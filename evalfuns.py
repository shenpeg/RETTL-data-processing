import pandas as pd
import stop_detection as sd 
import numpy as np
import copy
from tqdm import tqdm
try: 
    from collections import Iterable
except: 
    from collections.abc import Iterable
from tqdm import tqdm

# function definition 
def isWithinStop(row, stop): 

    """
    :param row: a pandas dataframe with only one row, has to have column named `time_stamp` 
    :param stop: a stop represented by a tuple, denoting the start and end timestamp of a stop
    :return: returns a boolean of whether this row's timestamp is within the given stop 
    """

    assert(type(stop) == tuple and len(stop) == 2) 

    stopStart, stopEnd = stop 
    rowTimestamp = row.loc["timestamp"]

    return stopStart <= rowTimestamp and rowTimestamp <= stopEnd


def getStopEvent(posDF, stops): 

    """
    This function generates the events and centroids of classroom actor's stops
    :param posDF: a pandas dataframe denoting the position data of an real-world object. Must have columns `chosen_X`, `chosen_Y`, and `time_stamp` 
    :param stops: an array of tuples denoting the start and end timestamps of stops. Usually values returned by sd.getStops()
    :return: returns a tuple of three arrays. The first is an array of strings, denoting the stopping events; the second 
             is a string of tuple points, denoting stop centroid coordinates; the third is a list of stop indeces to be 
             put into posDF. 
    """

    # values to be returned 
    events = [] 
    centroids = []
    stop_indeces = [] 

    i, j = 0, 0 # i is index for posDF; j is for indexing the list stops
    while(j < len(stops) ): 
        currStop = stops[j] 
        currStopStartInd = i
        inStop = False

        while(i < len(posDF) and isWithinStop(posDF.loc[i], currStop)): 
            inStop = True # indicate to the following code that we do run into the current stop 
            i += 1
            
        if(inStop): # did run into a stop, currStopInd and i shoud not be the same
            assert(currStopStartInd != i)

            rows = posDF.loc[currStopStartInd:i-1] # rows that are within currStop 
            points = sd.cols2tuples(rows.X, rows.Y) 
            centroid = sd.getCentroid(points) # get the centroid of current stop 
            assert(type(centroid) == tuple and len(centroid) == 2) # ensures that centroid is a point represented by a tuple

            stop_index = j 

            event = "Stopping in location: " + str(centroid) # format event in string
            
            # need to append multiple events since we are treating stop as a continuous event now
            for k in range(i - currStopStartInd):
                events.append(event) 
                centroids.append(centroid) 
                stop_indeces.append(stop_index)

            j += 1 # we have found all rows corresponding to the current stop, go to next stop

        else: # did not run into the current stop 
            assert(currStopStartInd == i)

            # not stopping event, denote as moving 
            events.append("Moving") 
            centroids.append(np.nan)
            stop_indeces.append(np.nan)

            # need to go to next row in position dataframe 
            i += 1
    
    # j reaches the end of stop list, but we still need to populate the events list to the same length as the original position dataframe 
    while(i < len(posDF)): 
        events.append("Moving") 
        centroids.append(np.nan)
        stop_indeces.append(np.nan)
        i += 1

    assert( len(events) == len(posDF) )
    assert( len(centroids) == len(posDF) )
    assert( len(stop_indeces) == len(posDF) )
            
    return events, centroids, stop_indeces


def getClosestObjs(actorDF, objDF, rng): 
    """
    This function returns a list of tuples, where the list is of the same length as actorDF. Tuples contain the names of top objects closest to the centroid stopping points specified in actorDF. Length of tuples are specified by numOfObjs parameter
    
    :param actorDF: pandas dataframe documenting the position of an `actor` by continuous unix timestamp. Must contain column `centroid` 
    :param objDF: pandas dataframe documenting the coordinates of all classroom objects. Must have columns: `object`, `X`, and `Y` 
    :param rng: range parameter; of any classroom objects is with the range distance of the stop centroid, this object gets thrown to the set of objects
    :return: returns a list of tuples. List is of the same length as actorDF. Tuples contain top objects closest to centroids of stops 
    """
    assert(type(rng) == int or type(rng) == float) 
    assert(rng > 0)

    closestObjs = [] # value to be returned, going to contain dictionaries in {<objName1>:<distance1>, <objName2>:<distance2>} format 
    centroids = actorDF.centroid # centroid points for stops, represented by tuples of two ints 
    #print(objDF)
    objPoints = sd.cols2tuples(objDF.X, objDF.Y) # X Y coordinates for classroom objects 
    objNames = objDF.object 
    assert(len(objNames) == len(objPoints)) # these two list/series should have one-to-one corresponding relation 

    i = 0 # indexing for centroids 
    while(i < len(centroids)): 

        centroid = centroids.iloc[i]
        if( np.any(np.isnan(centroid)) ): # actor is not in a stop 
            closestObjs.append(np.nan)

        elif(i - 1 >= 0 and centroids[i-1] == centroid): # if this current centroid is not the first one in the dataframe, and the previous centroid is the same as the current
            objDistDict = copy.deepcopy(closestObjs[len(closestObjs)-1]) # copy the previous object-distance dictionary
            closestObjs.append(objDistDict) # then append the copy

        else: # this means that we need to go through the coordinates of all the classroom objects to find these within range and append them to closetObjs list 
            objDistDict = dict() # create an empty dictionary to hold the entries in the future 
            
            j = 0
            while(j < len(objNames)): 
                if( sd.getDist(centroid, objPoints[j]) < rng ): # if object j within range
                    objDistDict[ objNames[j] ] = sd.getDist(centroid, objPoints[j]) # create a new entry as <object name>:<distance to centroid>
                j += 1

            closestObjs.append(objDistDict) 

        i += 1
        
    # To do for each entry in cloestObjs--> make sure to take closest actor

    assert(len(closestObjs) == len(actorDF)) # ensure that output length is correct 
    return(closestObjs)

def isEmpty(obj): 
    return not bool(obj)


def getObsInTimeframe(obsDF, timeframeStart, timeframeEnd): 
    obsTimestamps = obsDF["timestamp"] 
    # timestamps of observation data are ensured to be monotonically sorted 
    timeframeStartInd = obsTimestamps.searchsorted(timeframeStart) 
    timeframeEndInd = obsTimestamps.searchsorted(timeframeEnd) 
    return obsDF.loc[timeframeStartInd:timeframeEndInd] 

def calcTriangulationScoreAndPercentages(posDF, obsDF, n_stops, timeframe=10, id_prefix=''): 

    """
    :param posDF: distilled pozyx position data with stopping event and possible subjects specified 
    :param obsDF: distilled observation log data. See observation_distilled_sprint1_shou.tsv for an example 
    :param reward: reward points given when true subject in observaiton log appears in the set of possible subjects 
    :param penalty: penalty points deducted when incorrent subjects appear in possible subject set 
    :param timeframe: specified how many seconds we look back in time to find the correct subject, unit is second. 
    :return: returns the triangulation validation score for a combination of parameters; higher score means better alignment between modalities 
    """
    i_hits = i_false_alarms_inside = i_false_alarms_outside = i_misses = 0

    seen_stop_indeces = set() 

    i = 0  # indexing for observation data 
    
    while(i < len(obsDF)): 
        obsRow = obsDF.iloc[i] 
        obsEvent = obsRow["event"] # event name specified in observation data 

        # events that we can to valid with position data 
        if(obsEvent == "Talking to student: ON-task" or
           obsEvent == "Talking to student: OFF-task" or 
           obsEvent == "Talking to small group: ON-task" or 
           obsEvent == "Talking to small group: OFF-task" or
           obsEvent == "visit"): 

            trueSubjects = obsRow["subject"] # get the true subject(s) from observation data 
            assert(type(trueSubjects) == str and trueSubjects != "") # should now be a string but not empty, in format like "12;13"
            trueSubjects = trueSubjects.split(";") # split by semicolon since seat numbers are demilited by semicolons in distilling process 

            # we look both back and forward in time in position dataframe to check for occurrence of the true subject 
            back = timeframe / 2
            forward = timeframe - timeframe / 2
            assert(back + forward == timeframe)
            timeframeCenter = obsRow["timestamp"]
            timeframeStart = timeframeCenter - back 
            timeframeEnd = timeframeCenter + forward 
            # filter the position dataframe to get the rows within the timeframe and stopping 
            # TODO: Surpress warning here
            posInTimeframe = posDF[timeframeStart < posDF["timestamp"]][posDF["timestamp"] < timeframeEnd]
             
            subjSets = posInTimeframe["possibleSubjects"] 
            stop_indeces = posInTimeframe["stop_index"] 
            seen_stop_indeces.union( set(stop_indeces) )

            trueSubjects = [id_prefix+trueSubject for trueSubject in trueSubjects ]
            S = set(trueSubjects) # S is the target set here 
            
            
            G = set() # populate Guess set G 
            for g in subjSets: # g is the individual guess set here, will be merged into big G 
                if isinstance(g, dict): 
                    G = G.union(g)
                else: 
                    # this means that no guess is in the guess set g
                    assert np.isnan(g) 
                    
            hits = len( S.intersection(G) ) # number of hits = | S \intersect G | 
            misses = len( S - G ) 
            false_alarms = len( G - S )

            i_hits += hits 
            i_misses += misses 
            i_false_alarms_inside += false_alarms  

        i += 1

    # we are going to increment false alarms with the quantity below to prevent 
    # encouraging the algo to generate numerous false stops just for ramdom 
    # guessing
    for i in posDF.index: 
        event = posDF.loc[i, "event"]
        stop_index = posDF.loc[i, "stop_index"]
        # only check stops we did not check before hand 
        if ("Stopping" in event) and (stop_index not in seen_stop_indeces): 
            i_false_alarms_outside += len(posDF.loc[i, "possibleSubjects"])
            seen_stop_indeces.add(stop_index) # mark this stop as checked 

    return i_hits, i_misses, i_false_alarms_inside, i_false_alarms_outside

def visit_detection_based_on_position_data_routine(
    f_teacher = 'data/all_position_data_2021_observation_synced-cb-mar-24.csv',
    f_chart = 'data/seating_chart-cb-mar-24.csv',
    duration=21, radius=600, rng=700
    #duration=1, radius=60000, rng=1000
):
    
    # Read teacher file
    #teacherPos = pd.read_csv("demo_data/demo_position_data.csv", index_col=False) 
    teacherPos = pd.read_csv(f_teacher, index_col=False) 
    teacherPos.rename(columns={
        'time_stamp': 'timestamp',
        'chosen_X': 'X',
        'chosen_Y': 'Y'
    }, inplace=True)
    
    # Establish stops
    teacherStops = sd.getStops(teacherPos.X, teacherPos.Y, 
                                               teacherPos.timestamp, teacherPos.periodID, 
                                               teacherPos.dayID, duration, radius) 
    events, centroids, stop_indeces = getStopEvent(teacherPos, teacherStops) 

    teacherPos["event"] = events # populate event column for teacher positon dataframe 
    teacherPos["centroid"] = centroids 
    teacherPos["stop_index"] = stop_indeces 

    # Establish visits based on student position and new range parameter
    objPos = pd.read_csv(f_chart, index_col=False) 
    subjects = getClosestObjs(teacherPos, objPos, rng=rng)
    teacherPos["possibleSubjects"] = subjects 
    inferredSubj = getClosestObjs(teacherPos, objPos, rng)
    teacherPos["inferredSubjPos"] = inferredSubj
    
    return teacherPos

def evaluate_accuracy(teacherPos, f_obs='data/cleaned_obs_data-cb-mar-24.csv',
                     duration_ref=21, radius_ref=600, rng_ref=700, timeframe=10,
                      #duration_ref=2, radius_ref=5000, rng_ref=5000, timeframe=100,
                     bootstrap=False, obs_sep=',', id_prefix=''):
    obsLog = pd.read_csv(f_obs, index_col=False, sep=obs_sep) 
    
    if bootstrap:
        obsLog = obsLog.sample(obsLog.shape[0], replace=True).reset_index(drop=True).copy()
    
    calc = calcTriangulationScoreAndPercentages

    n_hits, n_misses, n_false_alarms_inside, \
    n_false_alarms_outside = calc(teacherPos, 
                                  obsLog, 
                                  len(teacherPos),
                                  timeframe=timeframe, id_prefix=id_prefix)
    
    precision = n_hits/(n_hits+n_false_alarms_outside) if (n_hits+n_false_alarms_outside) > 0 else np.nan
    recall = n_hits/(n_hits+n_misses) if (n_hits+n_misses) > 0 else np.nan
    
    
    newRow = {"duration": [duration_ref], 
              "radius": [radius_ref], 
              "range": [rng_ref], 
              "timeframe": [timeframe],
              'n_hits': [n_hits],
              'n_misses': [n_misses], 
              'n_false_alarms_inside': [n_false_alarms_inside], 
              'n_false_alarms_outside': [n_false_alarms_outside],
              'precision': [precision],
              'recall': [recall],
              'prec_rec_avg': [(precision+recall)/2]
             } 
    newDF = pd.DataFrame(newRow)
    
    return newDF

def bootstrap_evaluate_accuracy(teacherPos, f_obs='data/cleaned_obs_data-cb-mar-24.csv',
                     duration_ref=21, radius_ref=600, rng_ref=700, timeframe=10,
                     #duration_ref=2, radius_ref=5000, rng_ref=5000, timeframe=100,
                     n_samples=100):
    dfs = []
    for _ in tqdm(range(n_samples)):
        dfs.append(evaluate_accuracy(teacherPos, 
                                     f_obs=f_obs,
                     duration_ref=duration_ref, radius_ref=radius_ref, 
                                     rng_ref=rng_ref, timeframe=timeframe,
                                     bootstrap=True))
    df_eval = pd.concat(dfs)
    return df_eval

def proc_df_eval(df_eval):
    # Returns tuples of mean, lower, upper at 95% CIs
    precision = (round(df_eval.precision.mean(), 3), 
                 round(df_eval.precision.quantile(0.025), 3), 
                 round(df_eval.precision.quantile(0.975), 3))
    recall = (round(df_eval.recall.mean(), 3), 
              round(df_eval.recall.quantile(0.025), 3), 
              round(df_eval.recall.quantile(0.975), 3))
    prec_rec_avg = (round(df_eval.prec_rec_avg.mean(), 3), 
                    round(df_eval.prec_rec_avg.quantile(0.025), 3), 
                    round(df_eval.prec_rec_avg.quantile(0.975), 3))
    newRow = {
              'precision': [precision],
              'recall': [recall],
              'prec_rec_avg': [prec_rec_avg]
             } 
    newDF = pd.DataFrame(newRow)
    return newDF