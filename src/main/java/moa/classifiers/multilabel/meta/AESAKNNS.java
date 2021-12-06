package moa.classifiers.multilabel.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;

import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.Classifier;
import moa.classifiers.MultiLabelClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.multilabel.MLSAkNNSubspaces;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;

public class AESAKNNS extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	private static final long serialVersionUID = 1L;

	@Override
	public String getPurposeString() {
		return "Adaptive Ensemble of Self-Adjusting Nearest Neighbor Subspaces.";
	}    

	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", MLSAkNNSubspaces.class, "moa.classifiers.multilabel.MLSAkNNSubspaces");

	public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's', "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);
	
	public IntOption backgroundWindowSizeOption = new IntOption("backgroundWindowSize", 'y', "The number of instances in the background window.", 1000, 1, Integer.MAX_VALUE);

	protected Classifier[] ensemble;

	protected Classifier[] ensembleBackground;

	protected ADWIN[] ADError;

	protected long instancesSeen;
	protected long firstWarningOn;
	protected boolean warningDetected;
	
	@Override
	public void setModelContext(InstancesHeader context) {
		super.setModelContext(context);
		
		if(this.ensemble != null) {
			for (int i = 0; i < this.ensemble.length; i++) {
				this.ensemble[i].setModelContext(modelContext);
				this.ensembleBackground[i].setModelContext(modelContext);
			}
		}
	}

	@Override
	public void resetLearningImpl() {

		this.warningDetected = true;
		this.firstWarningOn = 0;
		this.instancesSeen = 0;

		this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
		this.ensembleBackground = new Classifier[this.ensembleSizeOption.getValue()];
		
		Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		if(modelContext != null) baseLearner.setModelContext(modelContext);
		baseLearner.resetLearning();

		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i] = baseLearner.copy();
		}

		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensembleBackground[i] = baseLearner.copy();
		}

		this.ADError = new ADWIN[this.ensemble.length];
		for (int i = 0; i < this.ensemble.length; i++) {
			this.ADError[i] = new ADWIN();
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		this.instancesSeen++;

		for (int i = 0; i < this.ensemble.length; i++) {
			
			Prediction prediction = this.ensemble[i].getPredictionForInstance(inst);
			((MLSAkNNSubspaces) this.ensemble[i]).evaluator.addResult(new InstanceExample(inst), prediction);
			
			double ErrEstim = this.ADError[i].getEstimation();
			
			boolean warning = false;

			// Update the drift detection method
			for(int o = 0; o < inst.numOutputAttributes(); o++) {
				if(prediction.getVotes(o) != null) {
					warning = this.ADError[i].setInput(Utils.maxIndex(prediction.getVotes(o)) == (int) inst.classValue(o) ? 0 : 1) || warning;
				}
			}
			
			if (warning && this.ADError[i].getEstimation() > ErrEstim) {
//				System.err.println("Change model "+i+"!");
				this.ensemble[i].resetLearning();
				this.ensemble[i].trainOnInstance(inst);
				this.ADError[i] = new ADWIN();
				
				if(this.warningDetected == false) {
					this.firstWarningOn = instancesSeen;
					this.warningDetected = true;
				}
			}
			
			int k = MiscUtils.poisson(1.0, this.classifierRandom);
			if (k > 0) {
				Instance weightedInst = (Instance) inst.copy();
				weightedInst.setWeight(inst.weight() * k);
				this.ensemble[i].trainOnInstance(weightedInst);
			}
		}

		if(this.warningDetected) {

			for (int i = 0; i < this.ensembleBackground.length; i++) {
				((MLSAkNNSubspaces) this.ensembleBackground[i]).evaluator.addResult(new InstanceExample(inst), this.ensembleBackground[i].getPredictionForInstance(inst));
				
				int k = MiscUtils.poisson(1.0, this.classifierRandom);
				if (k > 0) {
					Instance weightedInst = (Instance) inst.copy();
					weightedInst.setWeight(inst.weight() * k);
					this.ensembleBackground[i].trainOnInstance(weightedInst);
				}
			}

			if(this.instancesSeen - this.firstWarningOn == backgroundWindowSizeOption.getValue()) {
				// Compare the ensemble and the background ensemble. Select the best components
				for (int i = 0; i < this.ensembleBackground.length; i++) {

					double minSubsetAccuracyHamming = Double.MAX_VALUE;
					int minSubsetAccuracyHammingClassifier = -1;
					
					double tentativeSubsetAccuracy = ((MLSAkNNSubspaces) this.ensembleBackground[i]).evaluator.getSubsetAccuracy();
					double tentativeHammingScore = ((MLSAkNNSubspaces) this.ensembleBackground[i]).evaluator.getHammingScore();

					for (int j = 0; j < this.ensemble.length; j++) {
						
						double currentSubsetAccuracy = ((MLSAkNNSubspaces) this.ensemble[j]).evaluator.getSubsetAccuracy();
						double currentHammingScore = ((MLSAkNNSubspaces) this.ensemble[j]).evaluator.getHammingScore();
						
						if(currentSubsetAccuracy * currentHammingScore < minSubsetAccuracyHamming) {
							minSubsetAccuracyHamming = currentSubsetAccuracy * currentHammingScore;
							minSubsetAccuracyHammingClassifier = j;
						}
					}
					
					if(tentativeSubsetAccuracy * tentativeHammingScore > minSubsetAccuracyHamming) {
						this.ensemble[minSubsetAccuracyHammingClassifier] = this.ensembleBackground[i];
						this.ADError[minSubsetAccuracyHammingClassifier] = new ADWIN();
					}
				}

				Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
				baseLearner.setModelContext(this.modelContext);
				baseLearner.resetLearning();

				for (int i = 0; i < this.ensembleBackground.length; i++) {
					this.ensembleBackground[i] = baseLearner.copy();
				}

				this.warningDetected = false;
			}
		}
	}

	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance instance) {
		return getPredictionForInstance((new InstanceExample(instance)));
	}

	@Override
	public Prediction getPredictionForInstance(Example<Instance> example) {
		return compilePredictions(this.ensemble, example);
	}

	public static Prediction compilePredictions(Classifier h[], Example<Instance> example) {
		Prediction[] predictions = new Prediction[h.length];
		for (int i = 0; i < h.length; i++) {
			predictions[i] = h[i].getPredictionForInstance(example);
		}
		return combinePredictions(predictions, (Instance) example.getData());
	}

	public static Prediction combinePredictions(Prediction[] predictions, Instance inst) {
		Prediction result = new MultiLabelPrediction(inst.numOutputAttributes());
		for (int i = 0; i < predictions.length; i++) {
			try {
				Prediction more_votes = predictions[i];
				if (more_votes != null) {
					for (int numOutputAttribute = 0; numOutputAttribute < inst.numOutputAttributes(); numOutputAttribute++) {
						int length = 0;
						if (more_votes.getVotes(numOutputAttribute) != null)
							length = more_votes.getVotes(numOutputAttribute).length;
						for (int numValueAttribute = 0; numValueAttribute < length; numValueAttribute++) {
							result.setVote(numOutputAttribute, numValueAttribute,
									(result.getVote(numOutputAttribute, numValueAttribute) +
											more_votes.getVote(numOutputAttribute, numValueAttribute) / (double) predictions.length));
						}
					}
				}
			} catch (NullPointerException e) {
				System.err.println("NullPointer");
			} catch (ArrayIndexOutOfBoundsException e) {
				System.err.println("OutofBounds");
			}
		}
		return result;
	}

	//Legacy code: not used now, only Predictions are used
	@Override
	public double[] getVotesForInstance(Instance inst) {
		System.out.println("ERROR: double[] getVotesForInstance(Instance inst) SHOULD NOT BE CALLED");
		return null;
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance instance) {
		trainOnInstanceImpl((Instance) instance);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}
}