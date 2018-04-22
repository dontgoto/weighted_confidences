import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import uncertainties as unp
from uncertainties.unumpy import uarray as uarray
import glob

path = '/home/fongo/sync/bachelorarbeit/daten/rndforest/'

def calculate_status(prediction, label, weights=1):
    '''returns arrays containing total numbers of tp, tn, etc. for specific predictions, labels, and weights'''

    label = np.asarray(label).astype(bool)
    tp = prediction & label
    tn = ~prediction & ~label
    fp = prediction & ~label
    fn = ~prediction & label

    return np.sum(tp*weights), np.sum(tn*weights), np.sum(fp*weights), np.sum(fn*weights)


def calculate_for_cutoffs(confidence, label, weight=1, numberOfTrees = 100):
    '''returns arrays with length numberOfTrees, containing the sum of tp, tn, etc. for specific confidence cuts'''

    cutoff = np.linspace(0,1,numberOfTrees+1)
    predictions = [np.equal((confidence >= cut), True)
                   for cut in cutoff]
    values = np.array([calculate_status(prediction, label, weight)
                       for prediction in predictions])

    #return tp, tn, fp, fn
    return values[::,0], values[::,1], values[::,2], values[::,3]


def gen_classes(confidences, labels, weights=None, numberOfTrees=100):
    '''returns tp,tn,fp,fn for different confidence cuts : arrays of arrays'''
    if not np.any(weights[0]):
        weights = np.ones(len(confidences))

    values = np.array([calculate_for_cutoffs(confidence=confidence,
                                             label=label,
                                             weight=weight,
                                             numberOfTrees=numberOfTrees)
                        for confidence, label, weight in
                        zip(confidences, labels, weights)])
    # return tp, tn, fp, fn
	# the tps have the same dimension as the number of cutoffs
    return values[::,0], values[::,1], values[::,2], values[::,3]


def extract_data(globPath, weightKey='honda2014_spl_solmin.value', confidenceKey='confidence(true)', labelKey='label', sep=';'):
    '''Needs a shell glob string to find the csv files which get loaded into dataframes
    return values are arrays of arrays with the files as first dim and the values for the different events as second dim '''

    filenames = fileglobber(globPath)
    dataframes = [pd.read_csv(file, sep=sep)
                  for file in filenames]
    confidences = [np.array(dataframe[confidenceKey]) for dataframe in dataframes]
    labels = [np.array(dataframe[labelKey]) for dataframe in dataframes]
    if weightKey:
        weights = [np.array(dataframe[weightKey]) for dataframe in dataframes]
    else:
        weights = np.ones(len(labels))

    return confidences, labels, weights


def extract_data_and_calculate(globPath, **kwargs):
    '''Needs a shell glob string and returns tp,tn, etc. from confidence cuts
    return values are uarrays'''

    numberOfTrees = kwargs.pop('numberOfTrees', 100)

    confidences, labels, weights = extract_data(globPath)
    # confidences, labels, weights = extract_data(globPath, **kwargs)
    tp, tn, fp, fn = gen_classes(confidences,
                                 labels,
                                 weights,
                                 numberOfTrees=numberOfTrees)
    return tp, tn, fp, fn


def fileglobber(globPath):
    files = sorted(glob.glob(globPath))
    assert files, 'globPath doesnt point to valid filenames.'
    return files


def extract_Dataframe(globPath, keys, **kwargs):
    '''Needs a shell glob string to find the csv files which get loaded into dataframes
    return values are arrays of arrays with the files as first dim and the values for the different events as second dim
    takes a key array
    attrDict contains keys which are each matched to an array of arrays'''

    sep = kwargs.pop('sep', ';')
    # weightKey = keys.pop('weightKey', 'honda2014_spl_solmin.value')
    # confidenceKey = keys.pop('confidenceKey', 'confidence(true)')
    # labelKey = keys.pop('labelKey', 'label')
    dataframes = [pd.read_csv(file, sep=sep)
                  for file in fileglobber(globPath)]
    assert dataframes, 'Dataframes are empty'
    attrDict = {}

    for key in keys.values():
        attrDict[key] = [np.array(dataframe[key]) for dataframe in dataframes]
    assert attrDict, 'The dictionary is empty.'

    return attrDict


def confidence_histogram_from_csv(globPath, histTitle='', **kwargs):
    ''' creates a histgram with errorbars, showing the distribution of confidence values'''
    numberOfTrees = kwargs.get('numberOfTrees', 100)
    confidences, labels, weights = extract_data(globPath, **kwargs)
    # confidenceMean = [np.mean(i) for i in np.array(confidences).swapaxes(0,1)]
    # confidenceStd = [np.std(i) for i in np.array(confidences).swapaxes(0,1)]
    import pylab as plt
    plt.style.use('ggplot')
    plt.errorbar
    yAndBinEdges = [np.histogram(confidence,bins=numberOfTrees)
                 for confidence in confidences]
    yAndBinEdges = np.array(yAndBinEdges)
    ys = yAndBinEdges[::,0]
    ys = np.array([np.ndarray.tolist(y) for y in ys])
    binEdges = yAndBinEdges[0,1]
    yMean = [np.mean(ys[::,i]) for i in range(numberOfTrees)]
    yStd = [np.std(ys[::,i]) for i in range(numberOfTrees)]
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    width = 0.05
    plt.bar(bincenters, yMean, width=width, yerr=yStd)
    plt.xlim(0.0, 1.05)
    plt.title(histTitle)


def conf2(globPath, histTitle='', binRange=(0.0, 1.0), labelKey='label', defWeight=None,
          weightKey='honda2014_spl_solmin.value',
          keys={'true' : 'confidence(true)', 'false' : 'confidence(false)' },
          **kwargs):
    ''' returns mean and std binnings and binvalues for signal and background,
    showing the distribution of confidence values that can be plotted in a barplot etc. with errorbars
    of label=true and label=false separately, averaging the Eventnumbers per bin for all validations'''
    from matplotlib import pyplot as plt
    numberOfTrees = kwargs.get('numberOfTrees', 100)
    keys[labelKey] = labelKey
    keys[weightKey] = weightKey
    attrDict = extract_Dataframe(globPath, keys=keys, **kwargs)
    defaultKey = labelKey

    validationLength = len(attrDict[defaultKey])
    sigConfs = [attrDict[keys['true']][i][attrDict['label'][i] == True]
                for i in range(validationLength)]
    sigWeights = [attrDict[weightKey][i][attrDict['label'][i] == True]
                  for i in range(validationLength)]
    backConfs = [attrDict[keys['true']][i][attrDict['label'][i] == False]
                 for i in range(validationLength)]
    backWeights = [attrDict[weightKey][i][attrDict['label'][i] == False]
                   for i in range(validationLength)]

    if defWeight:
        for i in range(len(backWeights)):
            for j in range(len(backWeights[i])):
                backWeights[i][j] = 1.0
                sigWeights[i][j] = 1.0
    # backHistArray = gen_histarray(backConfs, backWeights, numberofTrees=numberOfTrees, range=binRange)
    backHistArray = np.array([plt.hist(backConf,
                                       bins=numberOfTrees,
                                       weights=backWeight,
                                       range=binRange)
                              for backConf,backWeight in
                              zip(backConfs, backWeights)])
    backNs = backHistArray[::,0]
    # backBins = backHistArray[::,1]
    sigHistArray = np.array([plt.hist(sigConf,
                                      bins=numberOfTrees,
                                      weights=sigWeight,
                                      range=binRange)
                             for sigConf,sigWeight in
                             zip(sigConfs, sigWeights)])
    sigNs = sigHistArray[::,0]
    sigBins = sigHistArray[::,1]

    sigMeanNs = np.mean(sigNs, axis=0)
    backMeanNs = np.mean(backNs, axis=0)
    sigStdNs = np.std(sigNs, axis=0)
    backStdNs = np.std(backNs, axis=0)
    plt.close()
    return sigMeanNs, sigStdNs, backMeanNs, backStdNs, sigBins[0]


def gen_histarray(confs, weights, numberOfTrees, binRange):
    '''confs and weights are arrays containing arrays for each validation'''
    histArray = np.array([plt.hist(conf, bins=numberOfTrees, weights=weight, range=binRange)
                     for conf,weight in zip(confs, weights)])
    return histArray


def gen_mean_std_quality(tp, tn, fp, fn, quantile=None):
    """takes arrays of arrays of tp, etc for different confidence cuts from a crossvalidations
    and returns uarrays of quality parameters for different confidence cuts
    optional quantile in percent from the median for replacing the standard deviations
    if quantile it will return an array with (mean, mean+quantile/2, mean-quantile/2)"""
    for arr in (tp, tn, fp, fn):
        arr = np.asarray(arr)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accMean = np.mean(accuracy, axis=0)
    recMean = np.mean(recall, axis=0)
    preMean = np.mean(precision, axis=0)
    if not quantile:
    # swapaxes to take the mean over the cross validations, doing a direct np.mean would lead to the average over different confidence cuts
        accStd = np.std(accuracy, axis=0)
        preStd = np.std(precision, axis=0)
        recStd = np.std(recall, axis=0)
        maximums = [np.max(mean+std) for mean,std in
                    zip([accMean, preMean, recMean], [accStd, preStd, recStd])]
        minimums = [np.min(mean-std) for mean,std in
                    zip([accMean, preMean, recMean], [accStd, preStd, recStd])]
        if (np.max(maximums) > 1.005) or (np.min(minimums) < -0.005):
            print('Warning: One of the quality parameters ± its stddev is out of the (-0.005,1.005) range. Using 84% quantiles instead.')
            quantile = 34
        else:
            return uarray(accMean, accStd), uarray(preMean, preStd), uarray(recMean, recStd)
    if quantile:
        accPlus = np.abs(np.percentile(accuracy, 50+(quantile/2), axis=0)- accMean)
        accMinus = np.abs(np.percentile(accuracy, 50-(quantile/2), axis=0) - accMean)
        prePlus = np.abs(np.percentile(precision, 50+(quantile/2), axis=0) - preMean)
        preMinus = np.abs(np.percentile(precision, 50-(quantile/2), axis=0) - preMean)
        recPlus = np.abs(np.percentile(recall, 50+(quantile/2), axis=0) - recMean)
        recMinus = np.abs(np.percentile(recall, 50-(quantile/2), axis=0) - recMean)
        return np.array([accMean, accMinus, accPlus]), np.array([ preMean, preMinus, prePlus]), np.array([ recMean, recMinus, recPlus])


def gen_quality_from_csv(globPath, sigFactor=None, backFactor=None, quantile=None, **kwargs):
    '''returns a dictionary containing the qualities for different confidence cuts, sigFactor and backFactor are factors needed when
    the separation is done on a subset of data and the total rates are to be calculated'''
    qualityNames = kwargs.pop('qualityNames', ('Accuracy', 'Reinheit', 'Effizienz'))
    tp, tn, fp, fn = extract_data_and_calculate(globPath, **kwargs)
    if sigFactor:
        tp *= sigFactor
        fn *= sigFactor
    if backFactor:
        fp *= backFactor
        tn *= backFactor
    qualityDict = {name : values for name,values in
                   zip(qualityNames, gen_mean_std_quality(tp, tn, fp, fn, quantile=quantile))}
    return qualityDict


def oneselection_plotter(qualityDict, numberOfTrees=100, quantile=False, fillBetween=False, **kwargs):
    '''creates an errorbar plot of with an arbitrary number of input
    elements from a dictionary of uarrays
    optional kwargs set plt.title(), defaults to '' '''

    nomval =unp.unumpy.nominal_values
    stdval = unp.unumpy.std_devs
    x = np.linspace(0, 1, numberOfTrees+1)
    title = kwargs.pop('title', '')
    plt.title(title)
    plt.ylabel('Rate')
    plt.ylim(0, 1.02)
    plt.xlabel('Confidence')
    elinewidth = kwargs.pop('elinewidth', 0.9)
    capsize = kwargs.pop('capsize', 2)
    markersize = kwargs.pop('markersize', 2)
    fmt = kwargs.pop('fmt', 'o')
    for arg in ['Reinheit', 'Effizienz']:
        if fillBetween is False:
            if quantile is False:
                plt.errorbar(x, nomval(qualityDict[arg]), fmt=fmt,
                            yerr=stdval(qualityDict[arg]), label=arg)
                            # elinewidth=elinewidth, capsize=capsize,
                            # markersize=markersize)
            else:
                plt.errorbar(x, qualityDict[arg][0], fmt=fmt,
                            yerr=[qualityDict[arg][1], qualityDict[arg][2]], label=arg)
                            # elinewidth=elinewidth, capsize=capsize,
                             # markersize=markersize)
        if fillBetween is True:
            if quantile is False:
                currentPlot = plt.plot(x, nomval(qualityDict[arg]), '-',
                                      label=arg)
                                       # markersize=markersize)
                currentColor = currentPlot[0].get_color()
                plt.fill_between(x, nomval(qualityDict[arg])-stdval(qualityDict[arg]),
                                 nomval(qualityDict[arg])+stdval(qualityDict[arg]),
                                 color=currentColor, alpha=0.3)
            else:
                currentPlot = plt.plot(x, qualityDict[arg][0], '-',
                                       label=arg)
                                       # markersize=markersize)
                currentColor = currentPlot[0].get_color()
                plt.fill_between(x, qualityDict[arg][0]-qualityDict[arg][1],
                                 qualityDict[arg][0]+qualityDict[arg][2],
                                 color=currentColor, alpha=0.3)

        #plt.plot(x, nomval(kwargs[arg]), color+'-', label=arg)
        #plt.fill_between(x, nomval(kwargs[arg])-stdval(kwargs[arg]), nomval(kwargs[arg])+stdval(kwargs[arg]), color=color, alpha=0.3)
    plt.legend(loc='best')


def values_from_ensembled_rndforests(confidences, label, weight):
    '''returns arrays with length numberOfTrees, containing the sum of tp, tn, etc. for specific confidence cuts
    the parameters need to be from forests which are fitted on different subsets of data, but applied on the same data'''

    treesPerForest = 15
    noOfTrees = treesPerForest*len(confidences)
    confidence = np.mean(confidences, axis=0)
    cutoff = np.linspace(0,1,noOfTrees+1)
    predictions = [np.equal((confidence >= cut), True)
                    for cut in cutoff]
    values = np.array([calculate_status(prediction, label, weight)
                       for prediction in predictions])
    #return tp, tn, fp, fn
    return values[::,0], values[::,1], values[::,2], values[::,3]


def extract_after_applying_model(h5signal, dataframeSig, extractKey, extractCol, confidenceCutoff=0.9, noOfTrees=100, vetoDiff=None):
    """`dataframeSig` is the .csv logfile written during the rndforest validation loop or when applying the
    rndforest on data, `h5signal` is a list of h5files containing the raw events on which the rndforest was
	applied.
    Vetodiff is a flag that sets if only the events that satisfy ext but not dc should be returned
    takes the events that are above a specific confidence cutoff and returns their reconstructed energies and
    true energies and corresponding event weights
    returns lists of `trueEnergies`, `recoEnergies` and `weights`"""

    assert ~dataframeSig.empty, 'dataframeSig is empty'
    assert h5signal, 'h5signal is empty'

    noOfnueFiles = 3
    noOfnumuFiles = 8
    runID = dataframeSig['Run'][dataframeSig['prediction(label)'] >= confidenceCutoff].astype(int)
    fileType = dataframeSig['fileType'][dataframeSig['prediction(label)'] >= confidenceCutoff]
    fileNumber = dataframeSig['fileNumber'][dataframeSig['prediction(label)'] >= confidenceCutoff]

    fileNumber[fileType == "nuMU"] += noOfnueFiles
    fileNumber[fileType == "corsika"] += noOfnueFiles + noOfnumuFiles

    attribute = []

    if extractKey == 'honda2014_spl_solmin':
        for i in range(noOfnueFiles):
            indices = runID[fileNumber == i]
            if vetoDiff:
                indices = indices[(h5signal[i]['DC_passed']['value'] == False) & (h5signal[i]['EXT_passed']['value'] == True)]
            attribute.extend(h5signal[i][extractKey][extractCol][indices])
        for i in range(noOfnueFiles, noOfnueFiles+noOfnumuFiles):
            indices = runID[fileNumber == i]
            if vetoDiff:
                indices = indices[(h5signal[i]['cc_in_deepcore']['value'] == False) & (h5signal[i]['cc_in_deepcore_ext']['value'] == True)]
            attribute.extend(h5signal[i]['weights']['honda2014_spl_solmin'][indices])
        for i in range(noOfnueFiles+noOfnumuFiles, len(h5signal)):
            indices = runID[fileNumber == i]
            if vetoDiff:
                indices = indices[(h5signal[i]['DC_passed']['value'] == False) & (h5signal[i]['EXT_passed']['value'] == True)]
            attribute.extend(np.ones(len(indices)) * 1/(7000*4.0143923434173443))
    else:
        for i in range(len(h5signal)):
            indices = runID[fileNumber == i]
            attribute.extend(h5signal[i][extractKey][extractCol][indices])

    return np.array(attribute)


def extract_trueE_recE(h5signal, dataframeSig, energyKey, confidenceCutoff=0.9, noOfTrees=100, defWeight=None):
    from functools import partial
    extractor = partial(extract_after_applying_model, h5signal, dataframeSig, confidenceCutoff=confidenceCutoff, noOfTrees=noOfTrees)
    trueEnergies = extractor('I3MCPrimary', 'energy')
    recoEnergies = extractor(energyKey, 'energy')
    if defWeight:
        weights = defWeight
    else:
        weights = extractor('honda2014_spl_solmin', 'value')

    return trueEnergies, recoEnergies, weights


# EXAMPLE USAGE
#corsikaNuCrosikaFeatures = gen_quality_from_csv(path+'*corsika_features*')
#oneselection_plotter(corsikaNuNuFeatures, title='largeRmrmr')


# things to put into config:
# keynames, plot defaults, quality names, separator in csv,
