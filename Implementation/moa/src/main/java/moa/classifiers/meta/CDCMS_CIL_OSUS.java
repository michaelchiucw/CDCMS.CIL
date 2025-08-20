/*
 *    CDCMS_CIL_OSUS.java
 *    Copyright (C) 2025 University of Birmingham, Birmingham, United Kingdom
 *    @author Chun Wai Chiu (michaelchiucw@gmail.com)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.diversitytest.QStatistics;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.driftdetection.DDM_GMean;
import moa.classifiers.core.driftdetection.DDM_OCI;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.Clusterer;
import moa.core.AutoClassDiscovery;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

public class CDCMS_CIL_OSUS extends AbstractClassifier implements MultiClassClassifier {

	/**
	 * Default serial version ID
	 */
	private static final long serialVersionUID = 1L;
	
	public IntOption randSeedOption = new IntOption("randomSeed", 'r',
            "Seed for random behaviour of the classifier.", 1);
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "trees.HoeffdingTree -l NB"); //trees.HoeffdingTree -e 2000000 -g 100 -c 0.01
	
	public IntOption poolSizeOption = new IntOption("ensembleSize", 'k',
			"The maximum size of the ensemble.", 10, 1, Integer.MAX_VALUE);
	
	public IntOption repositorySizeOption = new IntOption("repositorySizeMultiple", 'n',
			"The repository size will be n*k", 10, 1, Integer.MAX_VALUE);
	
	public IntOption timeStepsIntervalOption = new IntOption("timeStepsInterval", 'b',
			"The number of time steps after drift and before model recovery or between evaluation on ensemble_NL", 500, 1, Integer.MAX_VALUE);
	
	public FloatOption fadingFactorOption = new FloatOption("fadingFactor", 'f',
			"Fading Factor for prequential accuracy calculation on test chunk", 0.999, 0, 1);
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	public ClassOption descriptorsManagerOption = new ClassOption("descriptorsManager", 'm',
			"Clustering method to use as descriptors manager.", Clusterer.class, "clustream.Clustream");
	
	public FlagOption isUndersamplingDescriptorsOption = new FlagOption("isUndersamplingDescriptors", 'z', "isUndersamplingDescriptors?");
	
	public FloatOption similarityThresholdOption = new FloatOption("similarityThreshold", 's',
			"similarityThreshold", 0.8, 0.0, 1.0);
	
	public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'd',
            "Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");
	
	public FlagOption isUSOption = new FlagOption("isUS", 'u', "isUS?");
	
//	public ClassOption clustererOption = new ClassOption("clusterer", 'w',
//			"Clusterer for clustering models in repository.", Clusterer.class,
//			"WekaClusteringAlgorithm -w EM -p (-I 100 -N -1 -X 10 -max -1 -ll-cv 1.0E-6 -ll-iter 1.0E-6 -M 1.0E-6 -K 10 -num-slots 1 -S 100)");
	
	public MultiChoiceOption wekaAlgorithmOption;

	public StringOption parameterOption = new StringOption("parameter", 'p',
            "Parameters that will be passed to the weka algorithm. (e.g. '-N 5' for using SimpleKmeans with 5 clusters)",
            "");
			//-I 100 -N -1 -X 10 -max -1 -ll-cv 1.0E-6 -ll-iter 1.0E-6 -M 1.0E-6 -K 10 -num-slots 1 -S 100
	
    protected boolean isUndersamplingDescriptors;
	
	protected double similarityThreshold;
	
	protected EnsembleWithInfo ensemble_NL;
	
	protected EnsembleWithInfo ensemble_NH;
	
	protected EnsembleWithInfo ensemble_OL;
	
	protected ClassifierWithInfo candidate;
	
	protected List<ClassifierWithInfo> repository;
	protected int maxRepositorySize;
	
	protected ChangeDetector driftDetector;
	
	protected int afterDriftInstCount;
	
	private Instances predictionErrorByClassifierFromRepo;
	
	protected double warningDetected;
    protected double changeDetected;
    
    protected DRIFT_LEVEL previous_drift_level;
    protected DRIFT_LEVEL drift_level;
	
	private Class<?>[] clustererClasses;
	
	private weka.clusterers.AbstractClusterer clusterer;
	
	protected SamoaToWekaInstanceConverter instanceConverter;
	
	public CDCMS_CIL_OSUS() {
		this.clustererClasses = findWekaClustererClasses();
        String[] optionLabels = new String[clustererClasses.length];
        String[] optionDescriptions = new String[clustererClasses.length];

        for (int i = 0; i < this.clustererClasses.length; i++) {
            optionLabels[i] = this.clustererClasses[i].getSimpleName();
            optionDescriptions[i] = this.clustererClasses[i].getName();
        }

        if (this.clustererClasses != null && this.clustererClasses.length > 0) {
            wekaAlgorithmOption = new MultiChoiceOption("clusterer", 'w',
                    "Weka cluster algorithm to use.",
                    optionLabels, optionDescriptions, 2);
        } else {
            parameterOption = null;

        }
	}
	
	@Override
	public boolean isRandomizable() {
		return true;
	}
	
	@Override
	public void resetLearningImpl() {
		
		this.randomSeed = this.randSeedOption.getValue();
		this.classifierRandom = new Random(this.randomSeed);
		this.isUndersamplingDescriptors = this.isUndersamplingDescriptorsOption.isSet();
		
		// *-1, because more negative means more diverse in QStatistics.
		this.similarityThreshold = this.similarityThresholdOption.getValue() * -1;
		
		this.driftDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectorOption)).copy();
		
		this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
				((Clusterer) getPreparedClassOption(this.descriptorsManagerOption)).copy(), this.fadingFactorOption.getValue(),
				this.classifierRandom, this.isUndersamplingDescriptors);
		
		this.ensemble_NL = new EnsembleWithInfo(this.fadingFactorOption.getValue(), this.thetaOption.getValue(), this.isUSOption.isSet(), true, "NL");
		this.ensemble_NL.add(new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
				((Clusterer) getPreparedClassOption(this.descriptorsManagerOption)).copy(), this.fadingFactorOption.getValue(),
				this.classifierRandom, this.isUndersamplingDescriptors));
		
		this.ensemble_OL = null;
		this.ensemble_NH = null;
		
		this.maxRepositorySize = this.repositorySizeOption.getValue() * this.poolSizeOption.getValue();
		this.repository = new ArrayList<ClassifierWithInfo>(this.maxRepositorySize);
		
		this.afterDriftInstCount = 0;
		
		//====================================================================
		this.resetClusterer();
		
		this.instanceConverter = new SamoaToWekaInstanceConverter();
		
		this.changeDetected = 0;
        this.warningDetected = 0;
        this.previous_drift_level = DRIFT_LEVEL.NORMAL;
        this.drift_level = DRIFT_LEVEL.NORMAL;
        
	}
	
	private void initPredictionErrorStorage(int numAtt, int numberOfModels) {
		Attribute[] attributes = new Attribute[numAtt + 1];
		for (int i = 0; i < attributes.length - 1; ++i) {
			attributes[i] = new Attribute("Prediction " + (i+1));
		}
		attributes[attributes.length - 1] = new Attribute("Cluster Number");
		
		this.predictionErrorByClassifierFromRepo = new Instances("predictionErrorByClassifierFromRepo",
																 attributes,
																 numberOfModels);
		this.predictionErrorByClassifierFromRepo.setClassIndex(this.predictionErrorByClassifierFromRepo.numAttributes() - 1);
	}
	
	private void resetClusterer() {
		try {
            String clistring = clustererClasses[wekaAlgorithmOption.getChosenIndex()].getName();
            this.clusterer = (weka.clusterers.AbstractClusterer) ClassOption.cliStringToObject(clistring, weka.clusterers.Clusterer.class, null);

            String rawOptions = parameterOption.getValue();
            String[] options = rawOptions.split(" ");
            if (this.clusterer instanceof weka.core.OptionHandler) {
                ((weka.core.OptionHandler) this.clusterer).setOptions(options);
                Utils.checkForRemainingOptions(options);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	
	//TODO: For debugging:
	private void showPrequentialAccuracy() {
		
		double accuracySum = this.ensemble_NL.getPrequentialAccuracy() +
				(this.ensemble_NH == null ? 0.0 : this.ensemble_NH.getPrequentialAccuracy()) +
				(this.ensemble_OL == null ? 0.0 :this.ensemble_OL.getPrequentialAccuracy());
			
		System.out.println("proceed Instances: " + super.trainingWeightSeenByModel);
		
		if (this.ensemble_OL != null) {
			System.out.println("OL | Prequential Accuracy: " + this.ensemble_OL.getPrequentialAccuracy() + " | size: " + this.ensemble_OL.size() +
								" | weight: " + (this.ensemble_OL.getPrequentialAccuracy() / accuracySum));
		} else {
			System.out.println("OL | NULL");
		}
		
		if (this.ensemble_NH != null) {
			System.out.println("NH | Prequential Accuracy: " + this.ensemble_NH.getPrequentialAccuracy() + " | size: " + this.ensemble_NH.size() +
								" | weight: " + (this.ensemble_NH.getPrequentialAccuracy() / accuracySum));
		} else {
			System.out.println("NH | NULL");
		}
			
		if (this.ensemble_NL != null) {
			System.out.println("NL | Prequential Accuracy: " + this.ensemble_NL.getPrequentialAccuracy() + " | size: " + this.ensemble_NL.size() +
								" | weight: " + (this.ensemble_NL.getPrequentialAccuracy() / accuracySum));
		} else {
			System.out.println("NL | NULL");
		}
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		
		double[] to_return = null;
		
		double accuracy_NL = this.ensemble_NL.getPrequentialAccuracy();
		double accuracy_OL = this.ensemble_OL == null ? 0.0 : this.ensemble_OL.getPrequentialAccuracy();
		double accuracy_NH = this.ensemble_NH == null ? 0.0 : this.ensemble_NH.getPrequentialAccuracy();
		
		double accuracySum = accuracy_OL + accuracy_NH + accuracy_NL;
		
		DoubleVector combinedVote = new DoubleVector();
		
//		boolean a = false;
//		boolean b = false;
//		boolean c = false;
		
//		this.showPrequentialAccuracy();
		
		switch (this.drift_level) {
			case NORMAL:
				if (this.ensemble_OL != null && this.ensemble_NH != null &&
						accuracy_NL < accuracy_OL && accuracy_NL < accuracy_NH) {
					
					if (this.ensemble_OL.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_OL.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_OL / accuracySum);
							combinedVote.addValues(vote);
//							a = true;
						}
					}
					if (this.ensemble_NH.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_NH.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_NH / accuracySum);
							combinedVote.addValues(vote);
//							b = true;
						}
					}
					if (this.ensemble_NL.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_NL.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_NL / accuracySum);
							combinedVote.addValues(vote);
//							c = true;
						}
					}
					to_return = combinedVote.getArrayRef();

//					System.out.println("NORMAL A | OL: " + a + ", NH: " + b + ", NL: " + c + " | proceed Instances: " + super.trainingWeightSeenByModel);
//					this.showPrequentialAccuracy();
//					System.out.println("======================================");
					
				} else {
//					System.out.println("NORMAL B " + (this.ensemble_NL.estimation > 0.0 ? true : false) + " | proceed Instances: " + super.trainingWeightSeenByModel);
					to_return = this.ensemble_NL.getVotesForInstance(inst);
				}
				
					
				break;
			case OUTCONTROL:
				
				if (this.ensemble_OL.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_OL.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_OL / accuracySum);
						combinedVote.addValues(vote);
//						a = true;
					}
				}
				if (this.ensemble_NH.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_NH.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_NH / accuracySum);
						combinedVote.addValues(vote);
//						b = true;
					}
				}
				if (this.ensemble_NL.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_NL.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_NL / accuracySum);
						combinedVote.addValues(vote);
//						c = true;
					}
				}
				to_return = combinedVote.getArrayRef();
				
//				System.out.println("OUTCONTROL | OL: " + a + ", NH: " + b + ", NL: " + c + " | proceed Instances: " + super.trainingWeightSeenByModel);
//				this.showPrequentialAccuracy();
//				System.out.println("======================================");
				
				break;
			default:
				System.out.println("ERROR: getVotesForInstance()");
				break;
		}
//		if (to_return != null) {
//			System.out.print("Votes: [");
//			for (int i = 0; i < to_return.length; ++i) {
//				System.out.print(to_return[i] + " ");
//			}
//			System.out.print("]\n");
//		} else {
//			System.out.println("NULL");
//		}
//		
//		double[] singleVote = this.ensemble_NL.ensemble.get(0).getVotesForInstance(inst);
////		double sum = Arrays.stream(singleVote).sum(); 
//		System.out.print("Votes: [");
//		for (int i = 0; i < singleVote.length; ++i) {
//			System.out.print(singleVote[i]+ " ");
//		}
//		System.out.print("]\n");
		
		return to_return;
//		return this.ensemble_NL.getVotesForInstance(inst);
		
	}
	
	private void clusteringModels() throws Exception {
		weka.core.Instances wekaInstances = this.instanceConverter.wekaInstances(this.predictionErrorByClassifierFromRepo);
			
		weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
		filter.setAttributeIndices("" + (wekaInstances.classIndex() + 1));
				
		filter.setInputFormat(wekaInstances);
		weka.core.Instances wekaInstancesNoClass = weka.filters.Filter.useFilter(wekaInstances, filter);
				
		this.clusterer.buildClusterer(wekaInstancesNoClass);
				
		// Use forEachOrdered for debugging purposes.
		wekaInstancesNoClass.stream().forEach(wekaInstNoClass -> {
			try {
				int clusterLabel = this.clusterer.clusterInstance(wekaInstNoClass);
				int instIndex = wekaInstancesNoClass.indexOf(wekaInstNoClass);
				this.predictionErrorByClassifierFromRepo.get(instIndex).setClassValue(clusterLabel);
				if (instIndex < this.repository.size()) {
					this.repository.get(instIndex).setClusterLabel(clusterLabel);
				} else {
					this.ensemble_NL.ensemble.get(0).setClusterLabel(clusterLabel);
				}
						
//				System.out.print("(" + instIndex + ", " + clusterLabel + "), ");
//				if ((instIndex % 10 == 0 && instIndex != 0) || instIndex == wekaInstancesNoClass.size() - 1) {
//					System.out.println("");
//				}
			} catch (Exception e) {
//				System.out.println("instance #: " + this.trainingWeightSeenByModel);
//				System.out.println("wwekaInstancesNoClass.numAttributes(): " + wekaInstancesNoClass.numAttributes());
//				System.out.println("wekaInstancesNoClass.size(): " + wekaInstancesNoClass.size());
//				System.out.println("==============================================");
				e.printStackTrace();
			}
		});
	}
	
	private int getMostSimilarAndNewFromRepo(ClassifierWithInfo target) {
		
		if (this.repository.size() == 0) {
			return -1;
		}
		
		double[] qStatResults = new double[this.repository.size()];
		
		for (int i = 0; i < qStatResults.length; ++i) {
			List<ClassifierWithInfo> targetModels = Arrays.asList(target, this.repository.get(i));
			int targetNumOfDescriptors = this.getMostNumDescriptors(targetModels);
			List<Instance> testBatch = this.mergedAllCentreAsList(targetModels, targetNumOfDescriptors);
			qStatResults[i] = QStatistics.getQScoreForTwo(testBatch, target.getActualClassifier(), this.repository.get(i).getActualClassifier());
		}
		
		int maxQIndex = -1;
		for (int i = 0; i < qStatResults.length; ++i) {
			if (!Double.isNaN(qStatResults[i])) {
				maxQIndex = i;
				break;
			}
		}
		
		if (maxQIndex == -1) {
			return -1;
		}
		
		for (int i = maxQIndex + 1; i < qStatResults.length; ++i) {
			
			if (Double.isNaN(qStatResults[i])) {
				continue;
			}
			
			if (qStatResults[i] > qStatResults[maxQIndex]) {
				maxQIndex = i;
			} else if (qStatResults[i] == qStatResults[maxQIndex]) {
				maxQIndex = (this.repository.get(i).trainingWeightSeenByModel() 
								< this.repository.get(maxQIndex).trainingWeightSeenByModel()) ? i : maxQIndex;
			} else {
				/*
				 * Do nothing.
				 */
			}
		}
		
//		System.out.println("qStatResults[maxQIndex]: " + qStatResults[maxQIndex]);
		return qStatResults[maxQIndex] >= this.similarityThreshold ? maxQIndex : -1;
	}
	
	private int getMostNumDescriptors(List<ClassifierWithInfo> classifiers) {

		ArrayList<Integer> allSizes = new ArrayList<Integer>(classifiers.size() * 2);
		for (ClassifierWithInfo classifier : classifiers) {
			allSizes.add(classifier.getNumberOfDescriptors(0));
			allSizes.add(classifier.getNumberOfDescriptors(1));
		}
		
		return isUndersamplingDescriptors ? Collections.min(allSizes) : Collections.max(allSizes);
	}
	
	private List<Instance> mergedAllCentreAsList(List<ClassifierWithInfo> classifiers, int targetNumDescriptors) {
		List<Instance> mergedSet = new ArrayList<Instance>();
		for (ClassifierWithInfo classifier : classifiers) {
			if (classifier.getNumberOfDescriptors(0) > 0) {
				mergedSet.addAll(classifier.getDescriptorsCentre(0, targetNumDescriptors));
			}
			if (classifier.getNumberOfDescriptors(1) > 0) {
				mergedSet.addAll(classifier.getDescriptorsCentre(1, targetNumDescriptors));
			}
		}
		return mergedSet;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
		this.afterDriftInstCount++;
		
		double prediction = Utils.maxIndex(this.ensemble_NL.getVotesForInstance(inst)) == inst.classValue() ? 0.0 : 1.0;
//		this.driftDetector.input(prediction);
		/**
		 * DDM_OCI has to put before DDM_GMean because of polymorphism.
		 * DDM_OCI is a subclass of DDM_GMean
		 */
		if (this.driftDetector instanceof DDM_OCI) {
			((DDM_OCI) this.driftDetector).input(prediction, inst);
        } else if (this.driftDetector instanceof DDM_GMean) {
        	((DDM_GMean) this.driftDetector).input(prediction, inst);
        } else {
        	this.driftDetector.input(prediction);
        }
		
		this.drift_level = DRIFT_LEVEL.NORMAL;
		if (this.driftDetector.getChange()) {
			this.drift_level = DRIFT_LEVEL.OUTCONTROL;
		}
		
		switch (this.drift_level) {
			case NORMAL:
//				System.out.println("NORMAL");

				if (this.afterDriftInstCount == this.timeStepsIntervalOption.getValue() && this.changeDetected > 0 && this.repository.size() > 0) {
//					System.out.println("this.afterDriftInstCount: " + this.afterDriftInstCount);
					
					// determine the new model belongs to which cluster.
					// if CAN be determined: ensemble_NL = {C} ∪ {ClosestCluster.getModels(C, B)}
					// otherwise ensemble_NL = {C}
					
					int targetNumOfClusteringDescriptors = this.getMostNumDescriptors(this.repository);
					List<Instance> mergedSet = this.mergedAllCentreAsList(this.repository, targetNumOfClusteringDescriptors);
					this.initPredictionErrorStorage(mergedSet.size(), this.repository.size()+1);
					for (ClassifierWithInfo classifier : this.repository) {
						this.predictionErrorByClassifierFromRepo.add(classifier.makePredictionOnInstances(mergedSet));
					}
					this.predictionErrorByClassifierFromRepo.add(this.ensemble_NL.ensemble.get(0).makePredictionOnInstances(mergedSet));
			
					try {
						this.clusteringModels();
						
						int clusterToRecover = this.ensemble_NL.ensemble.get(0).getClusterLabel();
//						System.out.println("clusterToRecover: " + clusterToRecover);
//						
//						System.out.println("Before | ensemble_NL size: " + this.ensemble_NL.size());
						
						List<ClassifierWithInfo> sortedRepo = new ArrayList<ClassifierWithInfo>(this.repository);
						sortedRepo.sort(Comparator.comparing(ClassifierWithInfo::trainingWeightSeenByModel));
						
						for (ClassifierWithInfo classifier : sortedRepo) {
							if (this.ensemble_NL.size() >= this.poolSizeOption.getValue()) {
								break;
							}
							if (classifier.getClusterLabel() == clusterToRecover) {
								this.ensemble_NL.add(classifier);
							}
						}
						
//						for (ClassifierWithInfo classifier : this.repository) {
//							if (this.ensemble_NL.size() >= this.poolSizeOption.getValue()) {
//								break;
//							}
//							if (classifier.getClusterLabel() == clusterToRecover) {
//								this.ensemble_NL.add(classifier);
//							}
//						}
						
//						System.out.println("After | ensemble_NL size: " + this.ensemble_NL.size());
						
					} catch (Exception e) {
						e.printStackTrace();
					}
					
					this.predictionErrorByClassifierFromRepo.delete();
					this.resetClusterer();
					
					
				} else if (this.afterDriftInstCount % this.timeStepsIntervalOption.getValue() == 0 && this.trainingHasStarted()) {

					if (this.ensemble_NL.size() >= this.poolSizeOption.getValue()) {
						
						// Get the worst model from ensemble_NL.
						ClassifierWithInfo worstInNL = this.ensemble_NL.removeWorst();
						
						if (this.repository.size() >= this.maxRepositorySize) {
							
							int mostSimilarIndex = this.getMostSimilarAndNewFromRepo(worstInNL);
							
		 					if (mostSimilarIndex > -1 &&
		 							worstInNL.trainingWeightSeenByModel() > this.repository.get(mostSimilarIndex).trainingWeightSeenByModel()) {
		 						
		 						this.repository.remove(mostSimilarIndex);
								worstInNL.resetPrequentialAccuracy();
								this.repository.add(worstInNL);
								
							} else {
								/**
								 * Do nothing, worstInNL will then be discarded.
								 */
							}

						} else {
							worstInNL.resetPrequentialAccuracy();
							this.repository.add(worstInNL);
						}
						
						
					}
//					System.out.println("Ensemble Size: " + this.ensemble_NL.size() + " | Repo Size: " + this.repository.size());
					this.ensemble_NL.add(this.candidate);
//					System.out.println("Ensemble Size: " + this.ensemble_NL.size() + " | Repo Size: " + this.repository.size());
					
					this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
							((Clusterer) getPreparedClassOption(this.descriptorsManagerOption)).copy(), this.fadingFactorOption.getValue(),
							this.classifierRandom, this.isUndersamplingDescriptors);
					
				} else {
					this.candidate.updatePrequentialAccuracy(inst);
					this.candidate.trainOnInstance(inst);
				}
				
				this.previous_drift_level = DRIFT_LEVEL.NORMAL;
				
				break;
				
			case OUTCONTROL:
//				System.out.println("OUTCONTROL | " + super.trainingWeightSeenByModel);
				this.ensemble_OL = new EnsembleWithInfo(this.ensemble_NL);
				
				// Use NL because it will be clear afterwards, so can reset the prequential accuracy of the models without affecting OL 
				Boolean[] isAdd = new Boolean[this.ensemble_NL.size()];
				
				int tempMaxRepoSize = this.maxRepositorySize;
				
				for (int i = 0; i < this.ensemble_NL.size(); ++i) {
					
					if (this.repository.size() < tempMaxRepoSize) {
						isAdd[i] = true;
						--tempMaxRepoSize;
						continue;
					}
					
					int mostSimilarIndex = this.getMostSimilarAndNewFromRepo(this.ensemble_NL.getActualEnsemble().get(i));
					
					if (mostSimilarIndex > -1 &&
							this.ensemble_NL.getActualEnsemble().get(i).trainingWeightSeenByModel() > 
							this.repository.get(mostSimilarIndex).trainingWeightSeenByModel()) {
						
 						this.repository.remove(mostSimilarIndex);
 						isAdd[i] = true;
					} else {
						isAdd[i] = false;
					}
				}
				
				for (int i = 0; i < isAdd.length; ++i) {
					if (isAdd[i]) {
						ClassifierWithInfo toAdd = this.ensemble_NL.getActualEnsemble().get(i).copy();
						toAdd.resetPrequentialAccuracy();
						this.repository.add(toAdd);
					}
				}
				
				this.ensemble_NL.clear();
				
				this.ensemble_NH = new EnsembleWithInfo(this.fadingFactorOption.getValue(), this.thetaOption.getValue(), this.isUSOption.isSet(), false, "NH");
				
				if (this.previous_drift_level == DRIFT_LEVEL.NORMAL && this.repository.size() > 1) {
					this.candidate.resetLearning();
					
					// Do clustering
					// Create ensemble_NH
					
					int targetNumOfClusteringDescriptors = this.getMostNumDescriptors(this.repository);
					List<Instance> mergedSet = this.mergedAllCentreAsList(this.repository, targetNumOfClusteringDescriptors);
					this.initPredictionErrorStorage(mergedSet.size(), this.repository.size());
					for (ClassifierWithInfo classifier : this.repository) {
						this.predictionErrorByClassifierFromRepo.add(classifier.makePredictionOnInstances(mergedSet));
					}
						
					try {
						this.clusteringModels();
						
						int numOfClusters = this.clusterer.numberOfClusters();
						
						if (numOfClusters > 1) {
							
							// Get the most well-trained classifier from each cluster to form ensemble_NH.
							
							List<ClassifierWithInfo> sortedRepo = new ArrayList<ClassifierWithInfo>(this.repository);
							sortedRepo.sort(Comparator.comparing(ClassifierWithInfo::trainingWeightSeenByModel));
							
							
							for (int i = 0; i < sortedRepo.size() && numOfClusters > 0; ++i) {
								ClassifierWithInfo temp = sortedRepo.get(i);
//								System.out.println("have seen: " + temp.classifier.trainingWeightSeenByModel());
								if (temp.getClusterLabel() == numOfClusters - 1) {
									this.ensemble_NH.add(temp);
									i = 0;
									numOfClusters -= 1;
								}
								if (i == sortedRepo.size() - 1) {
									i = 0;
									numOfClusters -= 1;
								}
							}
							
						} else {
							for (int i = 0; i < this.repository.size() && this.ensemble_NH.size() < this.poolSizeOption.getValue(); i += this.repositorySizeOption.getValue()) {
								this.ensemble_NH.add(this.repository.get(i));
							}
						}
							
					} catch (Exception e) {
						e.printStackTrace();
					}
					
//					System.out.println("NH size: " + this.ensemble_NH.size());
//					for (int i = 0; i < this.ensemble_NH.size(); ++i ) {
//						System.out.println("index in R: " + this.repository.indexOf(this.ensemble_NH.ensemble.get(i)) 
//														  + " | Cluster label: " + this.ensemble_NH.ensemble.get(i).getClusterLabel());
//					}
//						
					this.predictionErrorByClassifierFromRepo.delete();
					this.resetClusterer();
				}
				
				this.ensemble_NL = new EnsembleWithInfo(this.fadingFactorOption.getValue(), this.thetaOption.getValue(), this.isUSOption.isSet(), true, "NL");
				this.ensemble_NL.add(candidate);
				
				this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
						((Clusterer) getPreparedClassOption(this.descriptorsManagerOption)).copy(), this.fadingFactorOption.getValue(),
						this.classifierRandom, this.isUndersamplingDescriptors);
				
				this.ensemble_NH.resetPrequentialAccuracy();
				this.ensemble_NL.resetPrequentialAccuracy();
				this.ensemble_OL.resetPrequentialAccuracy();
				
				this.afterDriftInstCount = 0;
				
				this.previous_drift_level = DRIFT_LEVEL.OUTCONTROL;
				
//				this.driftDetector.resetLearning();
				changeDetected++;
				
				break;
			default:
				System.out.print("ERROR!");
				break;
		}
		
		if (this.ensemble_OL != null) {
			this.ensemble_OL.updatePrequentialAccuracy(inst);
		}
		if (this.ensemble_NH != null) {
			this.ensemble_NH.updatePrequentialAccuracy(inst);
		}
		this.ensemble_NL.updatePrequentialAccuracy(inst);
		this.ensemble_NL.trainOnInstance(inst);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {

	}
	
	// Copied from WekaClusteringAlforithm.java
	private Class<?>[] findWekaClustererClasses() {
        AutoExpandVector<Class<?>> finalClasses = new AutoExpandVector<Class<?>>();
        Class<?>[] classesFound = AutoClassDiscovery.findClassesOfType("weka.clusterers",
                weka.clusterers.AbstractClusterer.class);
        for (Class<?> foundClass : classesFound) {
            finalClasses.add(foundClass);
        }
        return finalClasses.toArray(new Class<?>[finalClasses.size()]);
    }
	
	protected class EnsembleWithInfo extends AbstractClassifier {
		
		// TODO: For debugging
		private String name;
		
		private List<ClassifierWithInfo> ensemble;
		
		private double alpha;
		private double estimation;
		private double b;
		
		private boolean isWMEnsemble;
		
		private double[] classSizeEstimation;
		private double[] classSizeb;

		private double theta;
		
		private boolean isUS;
		
		protected EnsembleWithInfo(double alpha, double theta, boolean isUS, boolean isWMEnsemble, String name) {
			
			this.name = name;
			
			this.ensemble = new ArrayList<ClassifierWithInfo>();
			
			this.alpha = alpha;
			this.estimation = 0.0;
			this.b = 0.0;
			
			this.classSizeEstimation = null;
			this.classSizeb = null;

			this.theta = theta;
			
			this.isUS = isUS;
			
			this.isWMEnsemble = isWMEnsemble;
			
		}
		
		/*
		 * Copy Constructor
		 */
		protected EnsembleWithInfo(EnsembleWithInfo source) {
			
			if (source.name.equals("NL")) {
				this.name = "OL";
			}
			
			this.ensemble = new ArrayList<ClassifierWithInfo>(source.ensemble);
			
			this.alpha = source.alpha;
			this.estimation = source.estimation;
			this.b = source.b;
			
			this.classSizeEstimation = source.classSizeEstimation.clone();
			this.classSizeb = source.classSizeb.clone();
			
			this.theta = source.theta;
			
			this.isUS = source.isUS;
			
			this.isWMEnsemble = source.isWMEnsemble;
			
		}
		
		public EnsembleWithInfo copy() {
			return new EnsembleWithInfo(this);
		}
		
		protected int size() {
			return this.ensemble.size();
		}
		
		protected List<ClassifierWithInfo> getActualEnsemble() {
			return this.ensemble;
		}
		
		
		protected void clear() {
			this.ensemble.clear();
			this.estimation = 0.0;
			this.b = 0.0;
		}
		
		protected void add(ClassifierWithInfo toAdd) {
			this.ensemble.add(toAdd.copy());
		}
		
		protected ClassifierWithInfo removeWorst() {
			ClassifierWithInfo worst = this.ensemble
										   .stream()
										   .min(Comparator.comparingDouble(x -> x.getPrequentialAccuracy()))
										   .get();
			
//			for(int i = 0; i < this.ensemble.size(); ++i) {
//				System.out.println(i + ": " + this.ensemble.get(i).getPrequentialAccuracy() + " | have seen: " + this.ensemble.get(i).getTrainingWeightSeenByModel());
//			}
//			
//			System.out.println("worst: " + this.ensemble.indexOf(worst));
			this.ensemble.remove(worst);
			
			return worst;
		}
		
		public double[] getVotesForInstance(Instance inst) {
			
			double accuracySum = this.ensemble
									 .stream()
									 .mapToDouble(ClassifierWithInfo::getPrequentialAccuracy)
									 .sum();
			
			DoubleVector combinedVote = new DoubleVector();
			for (int i = 0; i < ensemble.size(); ++i) {
				if (ensemble.get(i).estimation > 0.0) {
					DoubleVector vote = new DoubleVector(ensemble.get(i).getVotesForInstance(inst));
						
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						if (isWMEnsemble) {
							vote.scaleValues(ensemble.get(i).getPrequentialAccuracy() / accuracySum);
						}
						combinedVote.addValues(vote);
					}
				}
			}
			
//			if (this.name.equals("NL") && drift_level == DRIFT_LEVEL.OUTCONTROL) {
//				System.out.println(this.name + ": combinedVote.sumOfValues(): " + combinedVote.sumOfValues() + " | instances: " + trainingWeightSeenByModel);
//			}
			
			return combinedVote.getArrayRef();
			
		}
		
		//-------------------------------OS/US methods----------------------------
		
		protected void updateClassSize(Instance inst) {
			if (this.classSizeEstimation == null) {
				classSizeEstimation = new double[inst.numClasses()];

				// <---start class size as equal for all classes
				for (int i=0; i<classSizeEstimation.length; ++i) {
					classSizeEstimation[i] = 1d/classSizeEstimation.length;
				}
			}
			if (this.classSizeb == null) {
				classSizeb = new double[inst.numClasses()];
				
				for (int i=0; i<classSizeEstimation.length; ++i) {
					classSizeb[i] = 1d/classSizeb.length;
				}
			}
			
			for (int i=0; i<classSizeEstimation.length; ++i) {
				classSizeEstimation[i] = thetaOption.getValue() * classSizeEstimation[i] + ((int) inst.classValue() == i ? 1d:0d);
				classSizeb[i] = thetaOption.getValue() * classSizeb[i] + 1d;
			}
		}
		
		protected double getClassSize(int classIndex) {
			return classSizeb[classIndex] > 0.0 ? classSizeEstimation[classIndex] / classSizeb[classIndex] : 0.0;
		}
		
		public double calculateWeightBaseOnClassSize(Instance inst) {
			double weight = 1d;
			int targetClass = this.isUS ? getMinorityClass() : getMajorityClass();
			
			weight = this.getClassSize(targetClass) / this.getClassSize((int) inst.classValue());

			return weight;
		}

		// will result in an error if classSize is not initialised yet
		public int getMajorityClass() {
			int indexMaj = 0;

			for (int i=1; i<classSizeEstimation.length; ++i) {
				if (this.getClassSize(i) > this.getClassSize(indexMaj)) {
					indexMaj = i;
				}
			}
			return indexMaj;
		}
		
		// will result in an error if classSize is not initialised yet
		public int getMinorityClass() {
			int indexMin = 0;

			for (int i=1; i<classSizeEstimation.length; ++i) {
				if (this.getClassSize(i) <= this.getClassSize(indexMin)) {
					indexMin = i;
				}
			}
			return indexMin;
		}
		
		//----------------------------------------------------------------------
		
		protected void updatePrequentialAccuracy(Instance inst) {
			this.estimation = this.alpha * this.estimation +
							(Utils.maxIndex(this.getVotesForInstance(inst)) == (int) inst.classValue() ? 1.0 : 0.0);
			
			this.b = this.alpha * this.b + 1.0;

			this.ensemble
				.stream()
				.forEach(committee -> committee.updatePrequentialAccuracy(inst));
		}
		
		protected double getPrequentialAccuracy() {
			return b > 0.0 ? this.estimation / this.b : 0.0;
		}
		
		protected void resetPrequentialAccuracy() {
			this.estimation = 0.0;
			this.b = 0.0;
			
			this.ensemble
				.stream()
				.forEach(committee -> committee.resetPrequentialAccuracy());
		}

		@Override
		public boolean isRandomizable() {
			return false;
		}

		@Override
		public void resetLearningImpl() {
			
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
			this.updateClassSize(inst);
			double weight = this.calculateWeightBaseOnClassSize(inst);
			
			Instance weightedInst = (Instance) inst.copy();
			weightedInst.setWeight(inst.weight() * weight);
			
			this.ensemble
				.stream()
				.forEach(committee -> committee.trainOnInstance(weightedInst));
			
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			
		}
		
	}
	
	protected class ClassifierWithInfo extends AbstractClassifier {
		
		private Classifier classifier;
		protected Clusterer[] descriptors;
		protected Random classifierRandom;
		protected boolean isUndersamplingDescriptors;
		
		private int clusterLabel;
		
		private double alpha;
		private double estimation;
		private double b;
		
		protected SamoaToWekaInstanceConverter moaToWekaInstanceConverter;
		protected WekaToSamoaInstanceConverter wekaToMoaInstanceConverter;
		
		Instances originalHeader;
		Instances nom2BinHeader;
		
		protected ClassifierWithInfo(Classifier classifier, Clusterer descriptorType, double prequentialAccFadingFactor, Random classifierRandom, boolean isUndersamplingDescriptors) {
			this.classifier = classifier;
			this.descriptors = new Clusterer[2]; // Assuming binary classification task
			for (int i = 0; i < this.descriptors.length; ++i) {
				this.descriptors[i] = descriptorType.copy();
			}
			this.classifierRandom = classifierRandom;
			this.isUndersamplingDescriptors = isUndersamplingDescriptors;
			this.alpha = prequentialAccFadingFactor;
			
			this.moaToWekaInstanceConverter = new SamoaToWekaInstanceConverter();
			this.wekaToMoaInstanceConverter = new WekaToSamoaInstanceConverter();
			
			this.resetLearning();
			
		}
		
		/*
		 * Copy Constructor
		 */
		protected ClassifierWithInfo(ClassifierWithInfo source) {
			this.classifier = source.classifier.copy();
			this.descriptors = source.descriptors.clone();
			this.classifierRandom = source.classifierRandom;
			this.isUndersamplingDescriptors = source.isUndersamplingDescriptors;
			
			this.clusterLabel = source.clusterLabel;
			
			this.alpha = source.alpha;
			this.estimation = source.estimation;
			this.b = source.b;
			
			this.moaToWekaInstanceConverter = source.moaToWekaInstanceConverter;
			this.wekaToMoaInstanceConverter = source.wekaToMoaInstanceConverter;
			
			this.originalHeader = source.originalHeader;
			this.nom2BinHeader = source.nom2BinHeader;
		}
		
		public ClassifierWithInfo copy() {
			return new ClassifierWithInfo(this);
		}
		
		@Override
		public double trainingWeightSeenByModel() {
			return this.classifier.trainingWeightSeenByModel();
		}
		
		protected Classifier getActualClassifier() {
			return this.classifier;
		}
		
		protected void setClusterLabel(int label) {
			this.clusterLabel = label;
		}
		
		protected int getClusterLabel() {
			return this.clusterLabel;
		}
		
		public double[] getVotesForInstance(Instance inst) {
			return this.classifier.getVotesForInstance(inst);
		}
		
		protected Instance makePredictionOnInstances(List<Instance> instances) {
			
			Instance predictions4Clustering = new DenseInstance(instances.size() + 1);
			
			predictions4Clustering.setDataset(predictionErrorByClassifierFromRepo);

			instances.stream()
					 .forEach(inst -> predictions4Clustering.setValue(instances.indexOf(inst),
														this.classifier.correctlyClassifies(inst) ? 1.0 : 0.0));			
			predictions4Clustering.setMissing(predictions4Clustering.classIndex());
			
			return predictions4Clustering;
		}
		
		protected void updatePrequentialAccuracy(Instance inst) {
			this.estimation = this.alpha * this.estimation + (this.classifier.correctlyClassifies(inst) ? 1.0 : 0.0);
			this.b = this.alpha * this.b + 1.0;
		}
		
		protected double getPrequentialAccuracy() {
			return b > 0.0 ? this.estimation / this.b : 0.0;
		}
		
		protected void resetPrequentialAccuracy() {
			this.estimation = 0.0;
			this.b = 0.0;
		}

		@Override
		public boolean isRandomizable() {
			return true;
		}

		@Override
		public void resetLearningImpl() {
			this.classifier.resetLearning();
			for (int i = 0; i < this.descriptors.length; ++i) {
				this.descriptors[i].resetLearning();
			}
			this.clusterLabel = -1;
			
			this.resetPrequentialAccuracy();
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
			if (this.originalHeader == null) {
				this.originalHeader = inst.dataset();
			}
			
			this.classifier.trainOnInstance(inst);
			Instance to_train_descriptor = inst.copy();
			try {
				to_train_descriptor = this.nominalToBinary(to_train_descriptor);
			} catch (Exception e) {
				e.printStackTrace();
			}
			to_train_descriptor.deleteAttributeAt(to_train_descriptor.classIndex());
			this.descriptors[(int) inst.classValue()].trainOnInstance(to_train_descriptor);
		}
		
		public List<Instance> getDescriptorsCentre(int classIndex, int targetNumberOfInstances) {
			Clustering mClusteringResult = this.descriptors[classIndex].getMicroClusteringResult();
			int numOfmClusters = mClusteringResult.size();
			AutoExpandVector<Cluster> mClusters = mClusteringResult.getClusteringCopy();
			
			/**
			 * Oversampling:
			 * Let the ratio = i + f, where i is an integer and f is a float number 0 <= f < 1.0.
			 * Let the number of descriptor of model C's class[classIndex] be N.
			 * Then:
			 * (1) add the whole set of the descriptor for i times. i.e. add N \times i instances.
			 * (2) add the remaining number of instances: targetNumberOfInstances - (N \times i)
			 * 
			 * UnderSampling:
			 * (3)
			 * while(targetSet.size() < targetNumberOfInstances) {
			 * 	targetSet add instance from currentSet with uniform probability.
			 * }
			 * 	[which eqauls to (2)]
			 */
			List<Instance> currentSet = new ArrayList<Instance>(numOfmClusters);
			List<Instance> targetSet = new ArrayList<Instance>(targetNumberOfInstances);
			int k = (int) Math.floor((targetNumberOfInstances * 1d) / (numOfmClusters * 1d));

			// Create currentSet:
			for (int i = 0; i < numOfmClusters; ++i) {
				// add class attribute and class value to the array.
				double[] centreWithClass = new double[mClusters.get(i).getCenter().length+1];
				System.arraycopy(mClusters.get(i).getCenter(), 0, centreWithClass, 0, mClusters.get(i).getCenter().length);
				centreWithClass[centreWithClass.length-1] = (double) classIndex;
				
				// Create an Instance based on the array; set weight to 1.0;
				Instance tempInst = new DenseInstance(1d, centreWithClass);
				tempInst.setDataset(this.nom2BinHeader);
				try {
					tempInst = this.binaryToNominal(tempInst, this.originalHeader);
				} catch (Exception e) {
					e.printStackTrace();
				}
				currentSet.add(tempInst);
			}
			
			if (!this.isUndersamplingDescriptors && k >= 1) {
				// (1):
				for (int i = 0; i < k; ++i) {
					targetSet.addAll(currentSet);
				}
			}
			// (2) or (3) :
			for (int i = 0; targetSet.size() < targetNumberOfInstances; ++i) {
				boolean isAdd = this.classifierRandom.nextBoolean();
				if (isAdd) {
					targetSet.add(currentSet.get(i % currentSet.size()));
				}
			}
			
			return targetSet;
		}
		
		public int getNumberOfDescriptors(int classIndex) {
			return this.descriptors[classIndex].getMicroClusteringResult().size();
		}
		
		private Instance binaryToNominal(Instance nom_inst, Instances original_header) throws Exception {
			
			Instances bin2nom_insts = new Instances(original_header);
			Instance tmp_inst = new DenseInstance(original_header.numAttributes());
			tmp_inst.setDataset(original_header);
			
			for (int i = 0; i < bin2nom_insts.numAttributes(); ++i) {
				String current_attr_name = bin2nom_insts.attribute(i).name();
				ArrayList<Double> values = new ArrayList<Double>();
				
				for (int j = 0; j < nom_inst.numAttributes(); ++j) {
					if (nom_inst.attribute(j).name().contains(current_attr_name)) {
						values.add(nom_inst.value(j));
					}
				}
				
				if (values.size() == 1) { // Numeric attrbute
					tmp_inst.setValue(i, values.get(0));
				} else if (values.size() > 1) { // Binary attribute: to be converted to nominal attribute
					
					double maxValue = Collections.max(values);
					int maxIndex = values.indexOf(maxValue);
					tmp_inst.setValue(i, maxIndex);
					
				} else { // matched_attributes.size == 0, which shouldn't happen but just in case.
					throw new Exception("No matched attribute.");
				}
			}
			bin2nom_insts.add(tmp_inst);
			
			return bin2nom_insts.get(0);
		}
		
		private Instance nominalToBinary(Instance original_inst) throws Exception {
			
			Instances moaInstances = new Instances(original_inst.dataset());
			moaInstances.add(original_inst);
			
			weka.core.Instances wekaInstances = this.moaToWekaInstanceConverter.wekaInstances(moaInstances);
			weka.filters.unsupervised.attribute.NominalToBinary nom2BinFilter = new weka.filters.unsupervised.attribute.NominalToBinary();
			nom2BinFilter.setInputFormat(wekaInstances);
			wekaInstances = weka.filters.Filter.useFilter(wekaInstances, nom2BinFilter);
			moaInstances = this.wekaToMoaInstanceConverter.samoaInstances(wekaInstances);
			if (this.nom2BinHeader == null) {
				this.nom2BinHeader = moaInstances.get(0).dataset();
			}
			
			return moaInstances.get(0);
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			
		}
	
	}
	
	protected enum DRIFT_LEVEL {
		NORMAL, WARNING, OUTCONTROL
	}

}
