package moa.classifiers.multilabel;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;
import moa.evaluation.PrequentialMultiLabelPerformanceEvaluator;

import java.util.*;

public class MLSAkNNSubspaces extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption maxWindowSize = new IntOption("maxWindowSize", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	public IntOption minWindowSize = new IntOption("minWindowSize", 'm', "The minimum number of instances to sotre",   50, 1, Integer.MAX_VALUE);

	public FloatOption penalty = new FloatOption("penalty", 'p', "Penalty ratio", 1, 0, Float.MAX_VALUE);

	public FloatOption reductionRatio = new FloatOption("reductionRatio", 'r', "Reduction ratio", 0.5, 0, 1);

	private String[] metrics = {"Subset Accuracy", "Hamming Score"};

	public MultiChoiceOption metric = new MultiChoiceOption("metric", 'e', "Choose metric used to adjust memory", metrics, metrics, 0);
	
	public IntOption kHistorySize = new IntOption("kHistorySize", 'k', "The history length for determining K value", 1000, 1, Integer.MAX_VALUE);
	
	public FloatOption percentageFeaturesMean = new FloatOption("percentageFeaturesMean", 'u', "Mean for percentage of featues selected", 0.7, 0, 1);
	
	public PrequentialMultiLabelPerformanceEvaluator evaluator;
	
	public boolean[] listAttributes;

	public InstancesHeader instanceHeader;

	private int numLabels;
	private int[] currentK;
	private List<Double>[][] KmetricHistory;
	private List<Instance> window;
	private double[][] distanceMatrix;
	private double[] attributeRangeMin;
	private double[] attributeRangeMax;
	private int[][] labelInstanceMask;
	private Map<Integer, List<Integer>> predictionHistories;
	private Map<Instance, Double> errors;
	
	@Override
	public String getPurposeString() {
		return "Multi-label Punitive kNN with Self-Adjusting Memory for Drifting Data Streams";
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		super.setModelContext(context);
		try {
			evaluator = new PrequentialMultiLabelPerformanceEvaluator();
			numLabels = context.numOutputAttributes();
			window = new ArrayList<Instance>();
			attributeRangeMin = new double[context.numInputAttributes()];
			attributeRangeMax = new double[context.numInputAttributes()];
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];
			predictionHistories = new HashMap<Integer, List<Integer>>();
			errors = new HashMap<Instance, Double>();
			labelInstanceMask = new int[maxWindowSize.getValue()][numLabels];

			currentK = new int[numLabels];
			for(int i = 0; i < numLabels; i++)
				currentK[i] = 3;

			KmetricHistory = new ArrayList[4][numLabels]; // 1, 3, 5, 7 per label
			for(int i = 0; i < 4; i++)
				for(int j = 0; j < numLabels; j++)
					KmetricHistory[i][j] = new ArrayList<Double>();

		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		if(window != null)
		{
			evaluator = new PrequentialMultiLabelPerformanceEvaluator();
			window.clear();
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];
			predictionHistories = new HashMap<Integer, List<Integer>>();
			errors = new HashMap<Instance, Double>();
			labelInstanceMask = new int[maxWindowSize.getValue()][numLabels];

			currentK = new int[numLabels];
			for(int i = 0; i < numLabels; i++)
				currentK[i] = 3;

			KmetricHistory = new ArrayList[4][numLabels]; // 1, 3, 5, 7 per label
			for(int i = 0; i < 4; i++)
				for(int j = 0; j < numLabels; j++)
					KmetricHistory[i][j] = new ArrayList<Double>();
		}
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance originalInst) {
		
		if(this.listAttributes == null) {
			setupListAttributes(originalInst);
		}
		
		MultiLabelInstance inst = (MultiLabelInstance) originalInst.copy();

		for(int att = inst.numInputAttributes()-1; att >= 0; att--) {
			if(this.listAttributes[att] == false) {
				inst.deleteAttributeAt(inst.numOutputAttributes() + att);	
			}
		}

		inst.setDataset(instanceHeader);

		window.add(inst);

		updateRanges(inst);

		for(int l = 0; l < numLabels; l++)
			labelInstanceMask[window.size()-1][l] = 1;

		int windowSize = window.size();

		get1ToNDistances(inst, window, distanceMatrix[windowSize-1]);

		List<Instance> discarded = new ArrayList<Instance>();

		for(Map.Entry<Instance, Double> entry : errors.entrySet())
		{
			if(entry.getValue() > penalty.getValue() * numLabels)
			{
				for(int idx = windowSize-1; idx >= 0; idx--)
				{
					if(window.get(idx) == entry.getKey())
					{
						for (int i = idx; i < windowSize-1; i++)
							for (int j = idx; j < i; j++)
								distanceMatrix[i][j] = distanceMatrix[i+1][j+1];

						for (int i = idx; i < windowSize-1; i++)
							labelInstanceMask[i] = labelInstanceMask[i+1];

						discarded.add(window.get(idx));
						window.remove(idx);
						windowSize--;
						break;
					}
				}
			}
		}

		for(Instance instance : discarded)
			errors.remove(instance);

		int newWindowSize = getNewWindowSize();

		if (newWindowSize < windowSize) {
			int diff = windowSize - newWindowSize;

			for (int i = 0; i < diff; i++)
				errors.remove(window.get(i));

			window = window.subList(diff, windowSize);

			for (int i = 0; i < newWindowSize; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[diff+i][diff+j];

			for (int i = 0; i < newWindowSize; i++)
				labelInstanceMask[i] = labelInstanceMask[diff+i];
		}

		if (newWindowSize == maxWindowSize.getValue()) {

			for (int i = 0; i < newWindowSize-1; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[i+1][j+1];

			for (int i = 0; i < newWindowSize-1; i++)
				labelInstanceMask[i] = labelInstanceMask[i+1];

			errors.remove(window.get(0));
			window.remove(0);
		}
	}

	/**
	 * Predicts the label of a given sample
	 */
	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance originalInstance) {
		
		if(this.listAttributes == null) {
			setupListAttributes(originalInstance);
		}
		
		MultiLabelInstance instance = (MultiLabelInstance) originalInstance.copy();

		for(int att = originalInstance.numInputAttributes()-1; att >= 0; att--) {
			if(this.listAttributes[att] == false) {
				instance.deleteAttributeAt(originalInstance.numOutputAttributes() + att);	
			}
		}

		instance.setDataset(instanceHeader);

		MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

		double[] distances = new double[window.size()];

		for (int i = 0; i < window.size(); i++)
			distances[i] = getDistance(instance, window.get(i));

		for(int j = 0; j < numLabels; j++)
		{
			double positives = 0;
			int totalVotes = 0;
			double totalVotesSum = 0;
			double[] distancesLocal = distances.clone();
			boolean modify = true;
			boolean[] added = new boolean[KmetricHistory.length];

			for(int n = 0; n < distancesLocal.length; n++) {

				double minDistance = Double.MAX_VALUE;
				int closestNeighbor = 0;

				for(int nn = 0; nn < distancesLocal.length; nn++) {
					if(distancesLocal[nn] <= minDistance) {
						minDistance = distancesLocal[nn];
						closestNeighbor = nn;
					}
				}

				distancesLocal[closestNeighbor] = Double.MAX_VALUE;

				boolean enter = false;

				if(labelInstanceMask[closestNeighbor][j] == 1) {
					if(window.get(closestNeighbor).classValue(j) == 1)
						positives += window.get(closestNeighbor).weight();

					totalVotes++;
					totalVotesSum += window.get(closestNeighbor).weight();
					enter = true;

					// If prediction was misleading, then disable the labelinstance
					if(modify && window.get(closestNeighbor).classValue(j) != instance.classValue(j)) {
						labelInstanceMask[closestNeighbor][j] = 0;

						Double instanceErrors = errors.remove(window.get(closestNeighbor));

						if(instanceErrors == null)
							errors.put(window.get(closestNeighbor), new Double(instance.weight()));
						else
							errors.put(window.get(closestNeighbor), instanceErrors.intValue() + instance.weight());
					}
				} else {
					// If labelinstance was disabled but it would had been a good prediction then reenable
					if(modify && window.get(closestNeighbor).classValue(j) == instance.classValue(j)) {
						labelInstanceMask[closestNeighbor][j] = 1;
					}
				}

				if((totalVotes == 1 || totalVotes == 3 || totalVotes == 5 || totalVotes == 7) && enter == true) {
					double relativeFrequency = positives / (double) totalVotesSum;
					int labelPrediction = relativeFrequency >= 0.5 ? 1 : 0; 
					int kIndex = -1;

					if(totalVotes == 1) {
						kIndex = 0;
					}
					else if(totalVotes == 3) {
						kIndex = 1;
					}
					else if(totalVotes == 5) {
						kIndex = 2;
					}
					else if(totalVotes == 7) {
						kIndex = 3;
					}

					if(labelPrediction == instance.classValue(j))
						KmetricHistory[kIndex][j].add(instance.weight());
					else
						KmetricHistory[kIndex][j].add(0.0);

					added[kIndex] = true;

					if(KmetricHistory[kIndex][j].size() > kHistorySize.getValue())
						KmetricHistory[kIndex][j].remove(0);

					if(totalVotes == currentK[j]) {
						prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
						modify = false;
					}

					if(totalVotes == 7) {
						break;
					}
				}
			}

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {
				if(added[kIndex] == false) {
					int labelPrediction = 0;
					
					if(labelPrediction == instance.classValue(j))
						KmetricHistory[kIndex][j].add(instance.weight());
					else
						KmetricHistory[kIndex][j].add(0.0);

					if(KmetricHistory[kIndex][j].size() > kHistorySize.getValue())
						KmetricHistory[kIndex][j].remove(0);
				}
			}
		}

		// Adapt current K to best accurate
		for(int j = 0; j < numLabels; j++) {

			double[] accuracy = new double[KmetricHistory.length];

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {

				double sum = 0;

				for(int v = 0; v < KmetricHistory[kIndex][j].size(); v++) {
					sum += KmetricHistory[kIndex][j].get(v);
				}

				accuracy[kIndex] = sum / (double) KmetricHistory[kIndex][j].size();
			}

			double bestAccuracyIndex = -1;
			double bestAccuracyValue = -1;

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {
				if(accuracy[kIndex] > bestAccuracyValue) {
					bestAccuracyValue = accuracy[kIndex];
					bestAccuracyIndex = kIndex;
				}
			}

			if(bestAccuracyIndex == 0) {
				currentK[j] = 1;
			}
			if(bestAccuracyIndex == 1) {
				currentK[j] = 3;
			}
			if(bestAccuracyIndex == 2) {
				currentK[j] = 5;
			}
			if(bestAccuracyIndex == 3) {
				currentK[j] = 7;
			}
		}

		return prediction;
	}

	private Integer getMetricSums(Instance instance, MultiLabelPrediction prediction) {
		int correct = 0;

		/** preset threshold */
		double t = 0.5;

		for (int j = 0; j < prediction.numOutputAttributes(); j++) {
			int yp = (prediction.getVote(j, 1) >= t) ? 1 : 0;
			correct += ((int) instance.classValue(j) == yp) ? 1 : 0;
		}

		return correct;
	}

	private double getMetricFromHistory(List<Integer> history) {

		double metric = 0.0;

		if(this.metric.getChosenLabel() == "Subset Accuracy")
		{
			for(Integer instanceSum : history)
				metric += (instanceSum == numLabels) ? 1 : 0;
		}
		else if (this.metric.getChosenLabel() == "Hamming Score")
		{
			for(Integer instanceSum : history)
				metric += instanceSum / (double) numLabels;
		}

		return metric / history.size();
	}

	/**
	 * Computes the Euclidean distance between one sample and a collection of samples in an 1D-array.
	 */
	private void get1ToNDistances(Instance sample, List<Instance> samples, double[] distances) {

		for (int i = 0; i < samples.size(); i++)
			distances[i] = getDistance(sample, samples.get(i));
	}

	/**
	 * Returns the Euclidean distance.
	 */
	private double getDistance(Instance instance1, Instance instance2) {

		double distance = 0;

		if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
		{
			for(int i = 0; i < instance1.numInputAttributes(); i++)
			{
				double val1 = instance1.valueInputAttribute(i);
				double val2 = instance2.valueInputAttribute(i);

				if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
				{
					val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					distance += (val1 - val2) * (val1 - val2);
				}
			}
		}
		else // Sparse Instance
		{
			int firstI = -1, secondI = -1;
			int firstNumValues  = instance1.numValues();
			int secondNumValues = instance2.numValues();
			int numAttributes   = instance1.numAttributes();
			int numOutputs      = instance1.numOutputAttributes();

			for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

				if (p1 >= firstNumValues) {
					firstI = numAttributes;
				} else {
					firstI = instance1.index(p1);
				}

				if (p2 >= secondNumValues) {
					secondI = numAttributes;
				} else {
					secondI = instance2.index(p2);
				}

				if (firstI < numOutputs) {
					p1++;
					continue;
				}

				if (secondI < numOutputs) {
					p2++;
					continue;
				}

				if (firstI == secondI) {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						double val2 = instance2.valueSparse(p2);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1 - val2) * (val1 - val2);
					}
					p1++;
					p2++;
				} else if (firstI > secondI) {
					int idx = secondI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val2 = instance2.valueSparse(p2);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val2) * (val2);
					}
					p2++;
				} else {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1) * (val1);
					}
					p1++;
				}
			}
		}

		return Math.sqrt(distance);
	}

	private void updateRanges(MultiLabelInstance instance) {
		for(int i = 0; i < instance.numInputAttributes(); i++)
		{
			if(instance.valueInputAttribute(i) < attributeRangeMin[i])
				attributeRangeMin[i] = instance.valueInputAttribute(i);
			if(instance.valueInputAttribute(i) > attributeRangeMax[i])
				attributeRangeMax[i] = instance.valueInputAttribute(i);
		}
	}

	/**
	 * Returns the bisected size which maximized the metric
	 */
	private int getNewWindowSize() {

		int numSamples = window.size();
		if (numSamples < 2 * minWindowSize.getValue())
			return numSamples;
		else {
			List<Integer> numSamplesRange = new ArrayList<Integer>();
			numSamplesRange.add(numSamples);
			while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * minWindowSize.getValue())
				numSamplesRange.add((int) (numSamplesRange.get(numSamplesRange.size() - 1) * reductionRatio.getValue()));

			Iterator<Integer> it = predictionHistories.keySet().iterator();
			while (it.hasNext()) {
				Integer key = (Integer) it.next();
				if (!numSamplesRange.contains(numSamples - key))
					it.remove();
			}

			List<Double> metricList = new ArrayList<Double>();
			for (Integer numSamplesIt : numSamplesRange) {
				int idx = numSamples - numSamplesIt;
				List<Integer> predHistory;
				if (predictionHistories.containsKey(idx))
					predHistory = getIncrementalTestTrainPredHistory(window, idx, predictionHistories.get(idx));
				else
					predHistory = getTestTrainPredHistory(window, idx);

				predictionHistories.put(idx, predHistory);

				metricList.add(getMetricFromHistory(predHistory));
			}
			int maxMetricIdx = metricList.indexOf(Collections.max(metricList));
			int windowSize = numSamplesRange.get(maxMetricIdx);

			if (windowSize < numSamples)
				adaptHistories(maxMetricIdx);

			return windowSize;
		}
	}

	/**
	 * Returns the n smallest indices of the smallest values (sorted).
	 */
	private int[] nArgMin(int n, double[] values, int startIdx, int endIdx, int label) {

		int indices[] = new int[n];

		for (int i = 0; i < n; i++){
			double minValue = Double.MAX_VALUE;
			for (int j = startIdx; j < endIdx + 1; j++){

				if (labelInstanceMask[j][label] == 1 && values[j] < minValue){
					boolean alreadyUsed = false;
					for (int k = 0; k < i; k++){
						if (indices[k] == j){
							alreadyUsed = true;
						}
					}
					if (!alreadyUsed){
						indices[i] = j;
						minValue = values[j];
					}
				}
			}
		}
		return indices;
	}

	public int[] nArgMin(int n, double[] values, int label) {
		return nArgMin(n, values, 0, values.length-1, label);
	}

	/**
	 * Returns the votes for each label.
	 */
	private double[] getPrediction(int[] nnIndices, List<Instance> instances, int j) {

		double count = 0;
		double sum = 0;

		for (int nnIdx : nnIndices) {
			if(instances.get(nnIdx).classValue(j) == 1) 
				count += instances.get(nnIdx).weight();
			
			sum += instances.get(nnIdx).weight();
		}

		double relativeFrequency = count / sum;

		return new double[]{1.0 - relativeFrequency, relativeFrequency};
	}

	/**
	 * Creates a prediction history from the scratch.
	 */
	private List<Integer> getTestTrainPredHistory(List<Instance> instances, int startIdx) {

		List<Integer> predictionHistory = new ArrayList<Integer>();

		for (int i = startIdx; i < instances.size(); i++) {

			MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

			for(int l = 0; l < numLabels; l++) {
				int nnIndices[] = nArgMin(Math.min(currentK[l], i - startIdx), distanceMatrix[i], startIdx, i-1 ,l);
				prediction.setVotes(l, getPrediction(nnIndices, instances, l));
			}

			predictionHistory.add(getMetricSums(instances.get(i), prediction));
		}

		return predictionHistory;
	}

	/**
	 * Creates a prediction history incrementally by using the previous predictions.
	 */
	private List<Integer> getIncrementalTestTrainPredHistory(List<Instance> instances, int startIdx, List<Integer> predictionHistory) {

		for (int i = startIdx + predictionHistory.size(); i < instances.size(); i++) {
			MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

			for(int l = 0; l < numLabels; l++) {
				int nnIndices[] = nArgMin(Math.min(currentK[l], distanceMatrix[i].length), distanceMatrix[i], startIdx, i-1, l);
				prediction.setVotes(l, getPrediction(nnIndices, instances, l));
			}

			predictionHistory.add(getMetricSums(instances.get(i), prediction));
		}

		return predictionHistory;
	}

	/**
	 * Removes predictions of the largest window size and shifts the remaining ones accordingly.
	 */
	private void adaptHistories(int numberOfDeletions) {
		for (int i = 0; i < numberOfDeletions; i++){
			SortedSet<Integer> keys = new TreeSet<Integer>(predictionHistories.keySet());
			predictionHistories.remove(keys.first());
			keys = new TreeSet<Integer>(predictionHistories.keySet());
			for (Integer key : keys){
				List<Integer> predHistory = predictionHistories.remove(key);
				predictionHistories.put(key-keys.first(), predHistory);
			}
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return true;
	}
	
	protected void setupListAttributes(MultiLabelInstance instance) {
		int numberAttributes = instance.numInputAttributes();
		
		int subspaceSize = (int) Math.round(this.percentageFeaturesMean.getValue() * numberAttributes + ((1.0 - this.percentageFeaturesMean.getValue()) * numberAttributes) * this.classifierRandom.nextGaussian() * 0.5);

		if (subspaceSize > numberAttributes) {
			subspaceSize = numberAttributes;
		} else if (subspaceSize <= 0) {
			subspaceSize = 1;
		}

		this.listAttributes = new boolean[numberAttributes];
		this.instanceHeader = new InstancesHeader(instance.dataset());
		
		ArrayList<Integer> attributesPool = new ArrayList<Integer>();

		for(int i = 0; i < numberAttributes; i++) {
			attributesPool.add(i);
		}

		for(int i = 0; i < subspaceSize; i++) {
			this.listAttributes[attributesPool.remove(this.classifierRandom.nextInt(attributesPool.size()))] = true;
		}

		for(int att = numberAttributes-1; att >= 0; att--) {
			if(this.listAttributes[att] == false) {
				this.instanceHeader.deleteAttributeAt(instance.numInputAttributes() + att);	
			}
		}
		
		this.instanceHeader.setClassIndex(Integer.MAX_VALUE); // FIXING ML CLASS INDEX AFTER DELETING ATTS
	}
}