package cotraining;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.sql.Date;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Random;


import cotraining.Entry;
import cotraining.Utilities;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author cornelia
 * 
 */

public class CSCoTraining_tT21 {

	
	public static CostSensitiveClassifier train(Instances train, Classifier classifier)throws Exception
    {
	           train.setClassIndex(train.numAttributes()-1);

            double []counts = new double[2];
            counts[0]=0;
            counts[1]=0;
            for (int tx=0; tx<train.numInstances(); tx++)
            {
                    if (train.instance(tx).classValue()==0)
                            counts[0]++;
                    else
                            counts[1]++;
            }
            
            CostMatrix cmatrix = new CostMatrix(2);
            cmatrix.setCell(0, 0, (double)0);
            cmatrix.setCell(1, 1, (double)0);
            cmatrix.setCell(0, 1, (double)1);
            cmatrix.setCell(1, 0, (double)1);
            
            if (counts[1]>counts[0]) //neg instances are more then cost of fn is more
            {
                    cmatrix.setCell(0, 1, (double)(counts[1]/counts[0]));
            }else if (counts[0]>counts[1])
            {
                    cmatrix.setCell(1, 0, (double)(counts[0]/counts[1]));
            }
             
            System.out.println("pos counts:"+counts[0]+" neg counts:"+counts[1]);
            System.out.println("Cost matrix "+cmatrix);
            
            CostSensitiveClassifier csclassifier = new CostSensitiveClassifier();
            csclassifier.setCostMatrix(cmatrix);
            csclassifier.setClassifier(classifier);
            
            try {
            	csclassifier.buildClassifier(train);
            } catch (Exception e1) {
                    // TODO Auto-generated catch block
                    e1.printStackTrace();
            }
            return csclassifier;
    }

	// the version with  ratios
	
	public static void testCoTraining(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int seed, int sample_size, int no_iter,
			String cType1, String cType2, 
			String output) throws Exception {
		
		//FileWriter fw = new FileWriter(output);
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		Random random;
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	
		int[] ratios = Utilities.getExpRatios(train_view1);
		//int[] ratios = {2,2};
		
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		
		//create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();
		
		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();
		
		//sampling u_prime1 and u_prime2
		random = new Random(seed);
		for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
			System.out.println(unlabeled_view2.numInstances());
			int index = random.nextInt(unlabeled_view2.numInstances());
			
			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);
			
			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} //done sampling u_prime1 and u_prime2
		
		//sum is the number of elements to be added to labeled data at each iteration	
		int sum = Utils.sum(ratios);
		int s = 2*sum; //the number of examples extracted at each iteration from each view
		//no_iter = (int) Math.floor((double)unlabeled_view_1.numInstances() / (double)s);
		//System.out.println("no_iter: "+no_iter);
			
		int crt_iter = 0; //counts "good" iterations - 
		  				  //when the right number of 
		  				  //instances can be added to labeled

		boolean enoughInstances = true; //makes sure that 
										//there are at least U' 
										//sample_size instances 
										//left in unlabeled

		double[] class_distr_1;
		double[] class_distr_2;

		//ea stands for entry array  
		//stores instances in U' and their predictions
		//used to make sorting easy 
		
		Entry[] ea_1 = null;
		Entry[] ea_2 = null;			
		
		//arrays of indices of instances 
		//that need to be added to the training
		ArrayList<Integer> al_1 = null;
		ArrayList<Integer> al_2 = null;
		
		//index_ea_i (i=1,2) has one entry for each class
		//tells us what's the last instance that we used 
		//in the list that will be added to labeled
		int[] index_ea_1 = null;
		int[] index_ea_2 = null;
		
		int[] indices_1 = null;
		int[] indices_2 = null;
		int[] indices = null;
		
		int len_1 = 0;
		int len_2 = 0;
		int len = 0;
			
		while((crt_iter < no_iter) && enoughInstances) {
			
			System.out.println("Number of labeled instances: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances: " + 
					labeled_view1.numInstances()+"\n\n");
			
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}

			
			
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
		
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
	        
			//foundFlag is true when instances in the right classes are found
			//among the predicted instances (in both views)
			
			boolean found1 = false;
			boolean found2 = false;
			
			boolean diff = true;
			
			while(diff && enoughInstances) {
				while((!found1 || !found2) && enoughInstances) {
					found1 = true;
					found2 = true;
						
					ea_1 = new Entry[u_prime1.numInstances()];
					ea_2 = new Entry[u_prime2.numInstances()];
						
					al_1 = new ArrayList<Integer>();
					al_2 = new ArrayList<Integer>();
					
					index_ea_1 = new int[labeled_view1.numClasses()];
					index_ea_2 = new int[labeled_view2.numClasses()];
						
					//classify u_prime1 with the classifier c1
					for(int j = 0; j < u_prime1.numInstances(); j ++) {
						
						Instance u_prime1_j = u_prime1.instance(j);
							
						class_distr_1 = csc1.distributionForInstance(u_prime1_j);
						double predicted = Utils.maxIndex(class_distr_1);
						String value = u_prime1.classAttribute().value((int)predicted);
							
						//add class probabilities together with instance indices to ea_1
						ea_1[j] = new Entry(j, class_distr_1[(int)predicted]);
							
						u_prime1_j.setClassMissing();
						u_prime1_j.setClassValue(value);
					}
						
					//ea_1 contains class probability and instance index 
					//sort ea_1 after class probabilities (increasing order)
					Arrays.sort(ea_1, 0, u_prime1.numInstances(), ea_1[0]);
					
					//find examples to add to the training set
					for(int j = 0; j < ratios.length; j ++){
						int n = ratios[j];
						int found = 0;
						
						//search in ea_1 from the end to the beginning
						for(int idx=ea_1.length-1; idx>=0; idx--){
							
							//get the index of the current instance (second comp in ea_1 - denoted by .i)
							int crt_idx = ea_1[idx].i;
								
							//get the instance with this best prediction and check its class
							Instance crt = u_prime1.instance(crt_idx);
							String cls = crt.stringValue(crt.classAttribute());
							
							//if cls is the class we are looking for
							if(cls.equals(labeled_view1.classAttribute().value(j))){
								//add the index of that instance to the array list
								al_1.add(crt_idx);
								found++;
									
								//if we don't need instances in this class anymore, 
								//remember where we left for that class
								if(found>=n){
									index_ea_1[j] = idx;
									break;
								}
							}
						}
						if (found<ratios[j]) {
							found1=false;
							break;
						}
					}
					
					//classify u_prime2 with the classifier c2
					for(int j = 0; j < u_prime2.numInstances(); j ++) {
						
						Instance u_prime2_j = u_prime2.instance(j);
						
						class_distr_2 = csc2.distributionForInstance(u_prime2_j);
						double predicted = Utils.maxIndex(class_distr_2);
						String value = u_prime2.classAttribute().value((int)predicted);
						
						//add class probabilities together with instance indices to ea_2
						ea_2[j] = new Entry(j, class_distr_2[(int)predicted]);
						
						u_prime2_j.setClassMissing();
						u_prime2_j.setClassValue(value);
			        }
							
					//sort ea_2 after class probabilities (increasing order)
					Arrays.sort(ea_2, 0, u_prime2.numInstances(), ea_2[0]);
						
					for(int j = 0; j < ratios.length; j ++){
						int n = ratios[j];
						int found = 0;
						
						for(int idx=ea_2.length-1; idx>=0; idx--){
							int crt_idx = ea_2[idx].i;
							
							Instance crt = u_prime2.instance(crt_idx);
							String cls = crt.stringValue(crt.classAttribute());
							if(cls.equals(labeled_view2.classAttribute().value(j))){
								
								al_2.add(crt_idx);
								found++;
								if(found>=n){
									index_ea_2[j] = idx;
									break;
								}
							}
						}
						if (found<ratios[j]) {
							found2=false;
							break;
						}
					}
					
					if (!found1 || !found2) {
						if (sample_size > unlabeled_view1.numInstances()) {
							enoughInstances = false;
						} else {
							//sample from unlabeled until double U'
							random = new Random(seed);
							for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
								int index = random.nextInt(unlabeled_view1.numInstances());
								
								Instance inst1 = unlabeled_view1.instance(index);
								u_prime1.add(inst1);
								unlabeled_view1.delete(index);
								
								Instance inst2 = unlabeled_view2.instance(index);
								u_prime2.add(inst2);
								unlabeled_view2.delete(index);
							}
						}
					}
				} //end while((!found1 || !found2) && enoughInstances)
			
				//sort indices to find out if there are instances that 
				//have conflicting labels in the two views
				len_1 = al_1.size();
				indices_1 = new int[len_1];
				for(int j = 0; j < len_1; j ++)
					indices_1[j] = al_1.get(j);
				Arrays.sort(indices_1, 0, len_1);
					
				len_2 = al_2.size();
				indices_2 = new int[len_2];
				for(int j = 0; j < len_2; j ++)
					indices_2[j] = al_2.get(j);
				Arrays.sort(indices_2, 0, len_2);
					
				len = len_1+len_2;
				indices = new int[len];
				for(int j = 0; j < len_1; j ++)
					indices[j] = indices_1[j];
				for(int j = 0; j < len_2; j ++)
					indices[j+len_1] = indices_2[j];
				Arrays.sort(indices, 0, len);
			
				diff = false;
				for(int j = len-1; j > 0; j--) {
					
					//check if an instance is in both lists
					if(indices[j] == indices[j-1]) {
											
						Instance crt_1 = u_prime1.instance(indices[j]);
						String cls_1 = crt_1.stringValue(crt_1.classAttribute());
						int cls_idx_1 = (int) crt_1.value(crt_1.classAttribute());
								
						Instance crt_2 = u_prime2.instance(indices[j]);
						String cls_2 = crt_2.stringValue(crt_2.classAttribute());
						int cls_idx_2 = (int) crt_2.value(crt_2.classAttribute());
	
						if (!cls_1.equals(cls_2)) {
									
							diff=true;
									
							//remove instances whose classes are different
							//from the two array lists al_1 and al_2
							al_1.remove((Integer)indices[j]);
							al_2.remove((Integer)indices[j]);
									
							found1 = false;
							found2 = false;
	
							//continue to search for another non-conflicting 
							//example from where we left and keep track of 
							//the new index in the corresponding class
								
							for(int idx=index_ea_1[cls_idx_1]-1; idx>=0; idx--) {
								int crt_idx = ea_1[idx].i;
										
								Instance crt = u_prime1.instance(crt_idx);
								String cls = crt.stringValue(crt.classAttribute());
								if(cls.equals(cls_1)) {
									al_1.add(crt_idx);
									index_ea_1[cls_idx_1]=idx;
									found1 = true;
									break;		
								}
							}
		
							for(int idx=index_ea_2[cls_idx_2]-1; idx>=0; idx--) {
								int crt_idx = ea_2[idx].i;
											
								Instance crt = u_prime2.instance(crt_idx);
								String cls = crt.stringValue(crt.classAttribute());
								if(cls.equals(cls_2)) {
									al_2.add(crt_idx);
									index_ea_2[cls_idx_2]=idx;
									found2 = true;
									break;
								}
							}
							
							if(!found1 || !found2)
								break;
						} //end if (!cls_1.equals(cls_2))	
						j=j-1;			
					}	
				}
				
				if (!found1 || !found2) {
					if (sample_size > unlabeled_view1.numInstances()) {
						enoughInstances = false;
					} else {
						random = new Random(seed);
						//sample from unlabeled until double U'
						for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
							int index = random.nextInt(unlabeled_view1.numInstances());
							
							Instance inst1 = unlabeled_view1.instance(index);
							u_prime1.add(inst1);
							unlabeled_view1.delete(index);
							
							Instance inst2 = unlabeled_view2.instance(index);
							u_prime2.add(inst2);
							unlabeled_view2.delete(index);
						}
					}
				}
			} //end while (diff) loop
			
			if (enoughInstances) {
                 //sort indices - we will delete them from the last to the first
				len_1 = al_1.size();
				indices_1 = new int[len_1];
				for(int j = 0; j < len_1; j ++)
					indices_1[j] = al_1.get(j);
				Arrays.sort(indices_1, 0, len_1);
				
				len_2 = al_2.size();
				indices_2 = new int[len_2];
				for(int j = 0; j < len_2; j++)
					indices_2[j] = al_2.get(j);
				Arrays.sort(indices_2, 0, len_2);
				
				len = len_1 + len_2;
				indices = new int[len];
				for(int j = 0; j < len_1; j ++)
					indices[j] = indices_1[j];
				for(int j = 0; j < len_2; j ++)
					indices[j+len_1] = indices_2[j];
				Arrays.sort(indices, 0, len);
	
				System.out.println("Done with iter "+crt_iter+
						" - adding examples for next iter");
				crt_iter++;
				//add best predicted in view 1
				System.out.println("Instances added to view2");
				for(int j = len_1-1; j >= 0; j--) {
					//find best predicted instances and their class in view 1
					Instance crt_1 = u_prime1.instance(indices_1[j]);
					String cls_1 = crt_1.stringValue(crt_1.classAttribute());
					//switch to view 2
					Instance crt_2 = u_prime2.instance(indices_1[j]);
						
					crt_2.setClassMissing();
					crt_2.setClassValue(cls_1);
						
					labeled_view2.add(crt_2);
					//labeled_view1.add(crt_1);
				}
				
				System.out.println("Instances added to view1");
				//add best predicted in view 2
				for(int j = len_2-1; j >= 0; j--) {
					//find best predicted instances and their class in view 2
					Instance crt_2 = u_prime2.instance(indices_2[j]);
					String cls_2 = crt_2.stringValue(crt_2.classAttribute());
					//switch to view 1
					Instance crt_1 = u_prime1.instance(indices_2[j]);
						
					crt_1.setClassMissing();
					crt_1.setClassValue(cls_2);
						
					labeled_view1.add(crt_1);
					//labeled_view2.add(crt_2);
				}
							
				//delete examples added to training from both views
				for(int j = len-1; j > 0; j--) {		
					if(indices[j] == indices[j-1]) {		
						u_prime1.delete(indices[j]);
						u_prime2.delete(indices[j]);
						j=j-1;			
					} else {
						u_prime1.delete(indices[j]);
						u_prime2.delete(indices[j]);
							
						if((j-1)==0) {		
							u_prime1.delete(indices[j-1]);
							u_prime2.delete(indices[j-1]);
						}
					}
				}
				
				//add 2*ratio examples to U' in both views
				if (s > unlabeled_view1.numInstances()) {
					enoughInstances = false;
				} else {
					//sample from unlabeled to add to U' s examples
					random = new Random(seed);
					for(int n_u = 0; n_u < s; n_u++) {
						int index = random.nextInt(unlabeled_view1.numInstances());
					
						Instance inst1 = unlabeled_view1.instance(index);
						u_prime1.add(inst1);
						unlabeled_view1.delete(index);
					
						Instance inst2 = unlabeled_view2.instance(index);
						u_prime2.add(inst2);
						unlabeled_view2.delete(index);
					}
				}
			}
		} // end of the iterations loop
				
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
        fw.close();        
       
	}
	
	public static void baseline_high(Instances train_view1, Instances test_view1, 
			Instances train_view2, Instances test_view2,
			String cType1, String cType2, String output) throws Exception {
		
		//FileWriter fw = new FileWriter(output);
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		Classifier basec1 = new SMO();
		Classifier basec2 = new SMO();
		
	    System.out.println("\nNumber of train instances: " + train_view1.numInstances());
	    System.out.println("Number of test instances: " + test_view1.numInstances());
	 	
	  	if(cType1.equals("NBM")){
			basec1 = new NaiveBayesMultinomial();				
		} else if (cType1.equals("NB")){
			basec1 = new NaiveBayes();
		} else if (cType1.equals("LR")){
			basec1 = new Logistic();
		} else if (cType1.equals("SVM")){
			basec1 = new SMO();
			((SMO) basec1).setBuildLogisticModels(true);
		} else if (cType1.equals("Bagging")){
			basec1 = new Bagging();
		} else if (cType1.equals("RF")){
			basec1 = new RandomForest();
		} else if (cType1.equals("AdaBoost")) {
			basec1 = new AdaBoostM1();
		} else {
			System.out.println("Classifier1 unknown!");
		}
		
		if(cType2.equals("NBM")){
			basec2 = new NaiveBayesMultinomial();
		} else if (cType2.equals("NB")){
			basec2 = new NaiveBayes();
		} else if (cType2.equals("LR")){
			basec2 = new Logistic();
		} else if (cType2.equals("SVM")){
			basec2 = new SMO();
			((SMO) basec2).setBuildLogisticModels(true);
		} else if (cType2.equals("Bagging")){
			basec2 = new Bagging();
		} else if (cType2.equals("RF")){
			basec2 = new RandomForest();
		} else if (cType2.equals("AdaBoost")) {
			basec2 = new AdaBoostM1();
		} else {
			System.out.println("Classifier2 unknown!");
		}

		
    	System.out.println("\nTrain!");
    	/*c1.buildClassifier(train_view1);
    	c2.buildClassifier(train_view2);
        */
    	CostSensitiveClassifier csc1 = train(train_view1, basec1);
    	CostSensitiveClassifier csc2 = train(train_view2, basec2);
        System.out.println("Test!");
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
        fw.close();
    	
	}
	
	public static void baseline_low(Instances train_view1, Instances test_view1, 
			Instances train_view2, Instances test_view2,
			String cType1, String cType2, double sizeF_l, 
			String output) throws Exception {
		
		//FileWriter fw = new FileWriter(output);
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		Classifier basec1 = new SMO();
		Classifier basec2 = new SMO();
		
	    System.out.println("\nNumber of train instances: " + train_view1.numInstances());
	    System.out.println("Number of test instances: " + test_view1.numInstances());
	  	
	  	
		Instances train_labeled1 = Utilities.getFractionLabeled(train_view1, sizeF_l, 0);
		Instances train_labeled2 = Utilities.getFractionLabeled(train_view2, sizeF_l, 0);
		
		

	  	if(cType1.equals("NBM")){
			basec1 = new NaiveBayesMultinomial();				
		} else if (cType1.equals("NB")){
			basec1 = new NaiveBayes();
		} else if (cType1.equals("LR")){
			basec1 = new Logistic();
		} else if (cType1.equals("SVM")){
			basec1 = new SMO();
			((SMO) basec1).setBuildLogisticModels(true);
		} else if (cType1.equals("Bagging")){
			basec1 = new Bagging();
		} else if (cType1.equals("RF")){
			basec1 = new RandomForest();
		} else if (cType1.equals("AdaBoost")) {
			basec1 = new AdaBoostM1();
		} else {
			System.out.println("Classifier1 unknown!");
		}
		
		if(cType2.equals("NBM")){
			basec2 = new NaiveBayesMultinomial();
		} else if (cType2.equals("NB")){
			basec2 = new NaiveBayes();
		} else if (cType2.equals("LR")){
			basec2 = new Logistic();
		} else if (cType2.equals("SVM")){
			basec2 = new SMO();
			((SMO) basec2).setBuildLogisticModels(true);
		} else if (cType2.equals("Bagging")){
			basec2 = new Bagging();
		} else if (cType2.equals("RF")){
			basec2 = new RandomForest();
		} else if (cType2.equals("AdaBoost")) {
			basec2 = new AdaBoostM1();
		} else {
			System.out.println("Classifier2 unknown!");
		}
		
    	System.out.println("\nTrain!");
    	/*c1.buildClassifier(train_labeled1);
    	c2.buildClassifier(train_labeled2);*/
        
    	CostSensitiveClassifier csc1 = train(train_labeled1, basec1);
    	CostSensitiveClassifier csc2 = train(train_labeled2, basec2);
    	
        System.out.println("Test!");

		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
        fw.close();
	}
	
	public static void main(String[] args) throws Exception {
		
        System.out.print(new Date(System.currentTimeMillis()) + "\t");
        System.out.print(new Time(System.currentTimeMillis()) + "\n");
		
		String path = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/allimage/";
		
		int no_iter = 50;
		
		int sample_size = 100;
		
	/*	double sizeF_l = 0.1;
		double sizeF_u = 0.5;*/
		
		int seed = 0;
		
		String cType1 = new String("NB");
		String cType2 = new String("SVM");
		
//		String m_Train1 = path + "/train_images.arff";
//		String m_Valid1 = path + "/valid1.arff";
//		String m_Test1 = path + "/test1.arff";
//		
//		String m_Train2 = path + "/train1.arff";
//		String m_Valid2 = path + "/valid1.arff";
//		String m_Test2 = path + "/test1.arff";
//		
//		
//		String m_Unlabeled1 = path + "/unlabeled1.arff";
//		String m_Unlabeled2 = path + "/unlabeled1.arff";
//		
	//	String output = path + "/output/ct_nbm_rf.txt";
		String output = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/results_sept6th/NBM_NBM_CrossAdd_CostInSensitive_BM.txt";
//		
//		System.out.println("\nLoading train_view1!");
//		DataSource source = null;
//	    source = new DataSource(m_Train1);
//	    Instances train_view1 = source.getDataSet();
//	    train_view1.setClassIndex(train_view1.numAttributes()-1);
//	    
//		
//	    System.out.println("\nLoading unlabeled_view1!");
//	    source = null;
//	    source = new DataSource(m_Unlabeled1);
//	    Instances unlabeled_view1 = source.getDataSet();
//	    unlabeled_view1.setClassIndex(unlabeled_view1.numAttributes()-1);
//	    
//	    System.out.println("\nLoading valid_view1!");
//	    source = null;
//	    source = new DataSource(m_Valid1);
//	    Instances valid_view1 = source.getDataSet();
//	    valid_view1.setClassIndex(valid_view1.numAttributes()-1);
//	    
//	    System.out.println("\nLoading test_view1!");
//	    source = null;
//	    source = new DataSource(m_Test1);
//	    Instances test_view1 = source.getDataSet();
//	    test_view1.setClassIndex(test_view1.numAttributes()-1);
//	    
//	    
//		System.out.println("\nLoading train_view2!");
//		source = null;
//	    source = new DataSource(m_Train2);
//	    Instances train_view2 = source.getDataSet();
//	    train_view2.setClassIndex(train_view2.numAttributes()-1);
//		
//	    System.out.println("\nLoading unlabeled_view2!");
//	    source = null;
//	    source = new DataSource(m_Unlabeled2);
//	    Instances unlabeled_view2 = source.getDataSet();
//	    unlabeled_view2.setClassIndex(unlabeled_view2.numAttributes()-1);
//	    
//	    System.out.println("\nLoading valid_view2!");
//	    source = null;
//	    source = new DataSource(m_Valid2);
//	    Instances valid_view2 = source.getDataSet();
//	    valid_view2.setClassIndex(valid_view2.numAttributes()-1);
//	    
//	    System.out.println("\nLoading test_view2!");
//	    source = null;
//	    source = new DataSource(m_Test2);
//	    Instances test_view2 = source.getDataSet();
//	    test_view2.setClassIndex(test_view2.numAttributes()-1);
	    ////////////////////////////////////
		System.out.println("loading TAGS!!!!");
		String m_Train1 = path + "/TAGS1.arff";
		String m_Test1 = path + "/TAGS2.arff";
		String m_Unlabeled1 = path + "/unlabeled_TAGS.arff";
		
		System.out.println("\nLoading train_view1!");
		DataSource source = null;
	    source = new DataSource(m_Train1);
	    Instances train_view1 = source.getDataSet();
	    train_view1.setClassIndex(train_view1.numAttributes()-1);
	    
		
	    System.out.println("\nLoading unlabeled_view1!");
	    source = null;
	    source = new DataSource(m_Unlabeled1);
	    Instances unlabeled_view1 = source.getDataSet();
	    unlabeled_view1.setClassIndex(unlabeled_view1.numAttributes()-1);
	    
	    System.out.println("\nLoading test_view1!");
	    source = null;
	    source = new DataSource(m_Test1);
	    Instances test_view1 = source.getDataSet();
	    test_view1.setClassIndex(test_view1.numAttributes()-1);
	    System.out.println("Finished loading TAGS!!!!");
		
	    ////////////////////////////////////
	    
	    System.out.println("Loading facial features");
	    String m_Train2 = path + "/RGB1.arff";
		System.out.println("\nLoading train_view2!");
		source = null;
	    source = new DataSource(m_Train2);
	    Instances train_view2 = source.getDataSet();
	    train_view2.setClassIndex(train_view2.numAttributes()-1);
		
	    String m_Train3 = path + "/EDGC1.arff";
		System.out.println("\nLoading train_view3!");
		source = null;
	    source = new DataSource(m_Train3);
	    Instances train_view3 = source.getDataSet();
	    train_view3.setClassIndex(train_view3.numAttributes()-1);
	    
	    String m_Train4=path + "/SIFT1.arff";
		System.out.println("\nLoading train_view4!");
		source = null;
	    source = new DataSource(m_Train4);
	    Instances train_view4 = source.getDataSet();
	    train_view4.setClassIndex(train_view4.numAttributes()-1);
	    
	    Instances trainViewFacial = joinInstances(train_view2, train_view3, train_view4);
	    trainViewFacial.setClassIndex(trainViewFacial.numAttributes()-1);
	    
	    ////////////////////////////////////
	    
	    ////////////////////////////////////
	    String m_Test2 = path + "/RGB2.arff";
		System.out.println("\nLoading test_view2!");
		source = null;
	    source = new DataSource(m_Test2);
	    Instances test_view2 = source.getDataSet();
	    test_view2.setClassIndex(test_view2.numAttributes()-1);
		
	    String m_Test3 = path + "/EDGC2.arff";
	    System.out.println("\nLoading test_view3!");
		source = null;
	    source = new DataSource(m_Test3);
	    Instances test_view3 = source.getDataSet();
	    test_view3.setClassIndex(test_view3.numAttributes()-1);
	    
	    String m_Test4= path + "/SIFT2.arff";
		System.out.println("\nLoading test_view4!");
		source = null;
	    source = new DataSource(m_Test4);
	    Instances test_view4 = source.getDataSet();
	    test_view4.setClassIndex(test_view4.numAttributes()-1);
	    
	    Instances testViewFacial = joinInstances(test_view2, test_view3, test_view4);
	    testViewFacial.setClassIndex(testViewFacial.numAttributes()-1);
	    
	    ////////////////////////////////////
	    String m_Unlabeled2 = path + "/unlabeled_RGB.arff";
		System.out.println("\nLoading unlabeled_view2!");
		source = null;
	    source = new DataSource(m_Unlabeled2);
	    Instances unlabeled_view2 = source.getDataSet();
	    unlabeled_view2.setClassIndex(unlabeled_view2.numAttributes()-1);
	    
	    
	    String m_Unlabeled3 = path + "/unlabeled_EDGC.arff";
		System.out.println("\nLoading unlabeled_view3!");
		source = null;
	    source = new DataSource(m_Unlabeled3);
	    Instances unlabeled_view3 = source.getDataSet();
	    unlabeled_view3.setClassIndex(unlabeled_view3.numAttributes()-1);
	    
	    String m_Unlabeled4 =path + "/unlabeled_SIFT.arff";
		System.out.println("\nLoading unlabeled_view4!");
		source = null;
	    source = new DataSource(m_Unlabeled4);
	    Instances unlabeled_view4 = source.getDataSet();
	    unlabeled_view4.setClassIndex(unlabeled_view4.numAttributes()-1);
	    
	    Instances unlabeledViewFacial = joinInstances(unlabeled_view2, unlabeled_view3, unlabeled_view4);
	    unlabeledViewFacial.setClassIndex(unlabeledViewFacial.numAttributes()-1);
	    
	    CSCoTraining_tT2.testCoTraining(trainViewFacial, null, train_view1, null, 
	    		unlabeledViewFacial, unlabeled_view1,
	    		testViewFacial, test_view1,
		   		seed, sample_size, no_iter, cType1, cType2, output);
	    
// CSCoTraining_tT2.testCoTraining(train_view1, valid_view1, train_view2, valid_view2, 
//	   		unlabeled_view1, unlabeled_view2,
//	   		test_view1, test_view2,
//	   		seed, sample_size, no_iter, cType1, cType2, output);

/*CSCoTraining_tT2.testCoTraining2(train_view1, valid_view1, train_view2, valid_view2, 
    		unlabeled_view1, unlabeled_view2, test_view1, test_view2, seed, sample_size, no_iter, cType1, cType2, output);
	*/ 
/*  CSCoTraining_tT2.testCoTrainingBlumMitchellNoRatio(train_view1, valid_view1, train_view2, valid_view2, 
	    		unlabeled_view1, unlabeled_view2, test_view1, test_view2, seed, sample_size, no_iter, cType1, cType2, output);

*/	  /*  CSCoTraining_tT2.testCoTraining(train_view1, test_view1, train_view2, test_view2, 
	    		seed, sample_size, sizeF_l, sizeF_u, no_iter, cType1, cType2, output);
	 */ /*
	    CSCoTraining_tT2.baseline_high(train_view1, test_view1, train_view2, test_view2, cType1, cType2, output);
	    
	    CSCoTraining_tT2.baseline_low(train_view1, test_view1, train_view2, test_view2, cType1, cType2, sizeF_l, output);
	 */   
		System.out.println("Done!\n");
		
        System.out.print(new Date(System.currentTimeMillis()) + "\t");
        System.out.println(new Time(System.currentTimeMillis()) + "\n");
		
	}


	private static Instances joinInstances(Instances view1,
			Instances view2, Instances view3) {
		Instances tempInstances = Instances.mergeInstances(view1, view2);
		return Instances.mergeInstances(tempInstances, view3);
	}


	static double confidence_high = 0.9;
	static double confidence_low = 1 - confidence_high;
	
	
	//cross - adding the most confident examples by each classifiers (conflicting labels not taken into account) 
	//with subsampling
	public static void testCoTraining2(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int seed, int sample_size, int no_iter,
			String cType1, String cType2, 
			String output) throws Exception {
		
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		Random random;		
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	   	
	  
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		
		//create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();
		
		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();
		
		//sampling u_prime1 and u_prime2
		random = new Random(seed);
		for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
			int index = random.nextInt(unlabeled_view1.numInstances());
			
			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);
			
			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} //done sampling u_prime1 and u_prime2
		
		//sum is the number of elements to be added to labeled data at each iteration	

		double[] class_distr_1;
		double[] class_distr_2;

		int crt_iter=0;
			
		while(crt_iter < no_iter && unlabeled_view1.numInstances()>0) {
			
			//crt_iter++;
			System.out.println("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances()+"\n\n");
			
			System.out.println("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances());
			fw.write("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances()+"\n\n");
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
			
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
			
			int numc1added=0;
			int numc2added=0;

			//classify u_prime1 with the classifier c1
			//if the probability is greater than, equal to the confidence-level 
			//add to the opposite set

			for(int j = 0; j < u_prime1.numInstances(); j ++) {

				Instance u_prime1_j = u_prime1.instance(j);							
				class_distr_1 = csc1.distributionForInstance(u_prime1_j);
				double predicted = Utils.maxIndex(class_distr_1);
				String value = u_prime1.classAttribute().value((int)predicted);

				//add class probabilities together with instance indices to ea_1
				if (class_distr_1[0] >= confidence_high || class_distr_1[0] <= confidence_low)
				{
					Instance temp = u_prime2.instance(j);
					temp.setClassMissing();
					temp.setClassValue(value);
					labeled_view2.add(temp);
					numc2added++;
				}						
			}

			//classify u_prime2 with the classifier c2
			for(int j = 0; j < u_prime2.numInstances(); j ++) {

				Instance u_prime2_j = u_prime2.instance(j);						
				class_distr_2 = csc2.distributionForInstance(u_prime2_j);
				double predicted = Utils.maxIndex(class_distr_2);
				String value = u_prime2.classAttribute().value((int)predicted);

				//add class probabilities together with instance indices to ea_2
				if (class_distr_2[0] >= confidence_high || class_distr_2[0] <= confidence_low)
				{
					Instance temp = u_prime1.instance(j);
					temp.setClassMissing();
					temp.setClassValue(value);
					labeled_view1.add(temp);
					numc1added++;
				}
			}

			System.out.println("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added);
			fw.write("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added+"\n");
			
			u_prime1.delete();
			u_prime2.delete();
			
			crt_iter++;

			//sample from unlabeled for a new sample of U'
			random = new Random(seed);
			for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
				int index = random.nextInt(unlabeled_view1.numInstances());

				Instance inst1 = unlabeled_view1.instance(index);
				u_prime1.add(inst1);
				unlabeled_view1.delete(index);

				Instance inst2 = unlabeled_view2.instance(index);
				u_prime2.add(inst2);
				unlabeled_view2.delete(index);
			}
		} // end of the iterations loop
				
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
		fw.close();
        
	}
	
	//adding the most confident examples by both classifiers (conflicting labels not taken into account) 
	//with subsampling
	public static void testCoTrainingBlumMitchellNoRatio(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int seed, int sample_size, int no_iter,
			String cType1, String cType2, 
			String output) throws Exception {
		
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		Random random;
		
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	   	
	  
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		
		//create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();
		
		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();
		
		//sampling u_prime1 and u_prime2
		random = new Random(seed);
		for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
			int index = random.nextInt(unlabeled_view1.numInstances());
			
			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);
			
			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} //done sampling u_prime1 and u_prime2
		
		//sum is the number of elements to be added to labeled data at each iteration	

		double[] class_distr_1;
		double[] class_distr_2;

		int crt_iter=0;
			
		while(crt_iter < no_iter && unlabeled_view1.numInstances()>0) {
			
			//crt_iter++;
			System.out.println("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances()+"\n\n");
			
			System.out.println("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances());
			fw.write("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances()+"\n\n");
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
			
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
			  				
			int numc1added=0;
			int numc2added=0;

			//classify u_prime1 with the classifier c1
			//if the probability is greater than, equal to the confidence-level 
			//add to the opposite set

			Hashtable<Integer, String> toadd_assigns = new Hashtable<Integer, String>();
			
			for(int j = 0; j < u_prime1.numInstances(); j ++) {

				Instance u_prime1_j = u_prime1.instance(j);							
				class_distr_1 = csc1.distributionForInstance(u_prime1_j);
				double predicted = Utils.maxIndex(class_distr_1);
				String value = u_prime1.classAttribute().value((int)predicted);
				
				double max=class_distr_1[0];
				for (int tx=1; tx<class_distr_1.length; tx++)
				{
					if (class_distr_1[tx]>max)
					{
						max = class_distr_1[tx];
					}
				}
				if (max >= confidence_high || max <= confidence_low)
				{	
					toadd_assigns.put(j, value);
				}
			}

			//classify u_prime2 with the classifier c2
			for(int j = 0; j < u_prime2.numInstances(); j++) {

				Instance u_prime2_j = u_prime2.instance(j);						
				class_distr_2 = csc2.distributionForInstance(u_prime2_j);
				double predicted = Utils.maxIndex(class_distr_2);
				String value = u_prime2.classAttribute().value((int)predicted);
				
				double max=class_distr_2[0];
				for (int tx=1; tx<class_distr_2.length; tx++)
				{
					if (class_distr_2[tx]>max)
					{
						max = class_distr_2[tx];
					}
				}				
				
				String value1 = toadd_assigns.get(j);
				
				if (max >= confidence_high || max <= confidence_low)
				{	
					if (value1==null)
					{
						toadd_assigns.put(j, value);
					}
				}							
			}
			
			for(int j = 0; j < u_prime2.numInstances(); j++) {
				
				String value1 = toadd_assigns.get(j);
				if (value1==null)
					continue;
		
				Instance temp = u_prime1.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view1.add(temp);
				numc1added++;

				temp = u_prime2.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view2.add(temp);
				numc2added++;					

			}
			System.out.println("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added);
			fw.write("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added+"\n");
			
			u_prime1.delete();
			u_prime2.delete();
			
			crt_iter++;

			//sample from unlabeled for a new sample of U'
			random = new Random(seed);
			for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
				int index = random.nextInt(unlabeled_view1.numInstances());

				Instance inst1 = unlabeled_view1.instance(index);
				u_prime1.add(inst1);
				unlabeled_view1.delete(index);

				Instance inst2 = unlabeled_view2.instance(index);
				u_prime2.add(inst2);
				unlabeled_view2.delete(index);
			}
		} // end of the iterations loop
				
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
		fw.close();
        
	}

	
	//adding the most confident examples by both classifiers (no conflicting labels) 
	// with subsampling
	public static void testCoTrainingBlumMitchellNoRatio2(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int seed, int sample_size, int no_iter,
			String cType1, String cType2, 
			String output) throws Exception {
		
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		Random random;
		
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	   	
	  
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		
		//create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();
		
		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();
		
		//sampling u_prime1 and u_prime2
		random = new Random(seed);
		for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
			int index = random.nextInt(unlabeled_view1.numInstances());
			
			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);
			
			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} //done sampling u_prime1 and u_prime2	

		double[] class_distr_1;
		double[] class_distr_2;

		int crt_iter=0;
			
		while(crt_iter < no_iter && unlabeled_view1.numInstances()>0) {
			
			//crt_iter++;
			System.out.println("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances()+"\n\n");
			
			System.out.println("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances());
			fw.write("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances()+"\n\n");
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
			
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
			  				
			int numc1added=0;
			int numc2added=0;

			//classify u_prime1 with the classifier c1
			//if the probability is greater than, equal to the confidence-level 
			//add to the opposite set

			Hashtable<Integer, String> toadd_assigns = new Hashtable<Integer, String>();
			
			for(int j = 0; j < u_prime1.numInstances(); j ++) {

				Instance u_prime1_j = u_prime1.instance(j);							
				class_distr_1 = csc1.distributionForInstance(u_prime1_j);
				double predicted = Utils.maxIndex(class_distr_1);
				String value = u_prime1.classAttribute().value((int)predicted);
				
				double max=class_distr_1[(int)predicted];
				if (max >= confidence_high)
				{	
					toadd_assigns.put(j, value);
				}
			}

			//classify u_prime2 with the classifier c2
			for(int j = 0; j < u_prime2.numInstances(); j++) {

				Instance u_prime2_j = u_prime2.instance(j);						
				class_distr_2 = csc2.distributionForInstance(u_prime2_j);
				double predicted = Utils.maxIndex(class_distr_2);
				String value = u_prime2.classAttribute().value((int)predicted);
				
				double max=class_distr_2[(int)predicted];				
				
				String value1 = toadd_assigns.get(j);
				
				if (max >= confidence_high)
				{	
					if (value1==null)
					{
						toadd_assigns.put(j, value);
					} else {
						if(!value1.equals(value))
							toadd_assigns.remove(j);
					}
				}							
			}
			
			for(int j = 0; j < u_prime2.numInstances(); j++) {
				
				String value1 = toadd_assigns.get(j);
				if (value1==null)
					continue;
		
				Instance temp = u_prime1.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view1.add(temp);
				numc1added++;

				temp = u_prime2.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view2.add(temp);
				numc2added++;					

			}
			System.out.println("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added);
			fw.write("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added+"\n");
			
			u_prime1.delete();
			u_prime2.delete();
			
			crt_iter++;

			//sample from unlabeled for a new sample of U'
			random = new Random(seed);
			for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
				int index = random.nextInt(unlabeled_view1.numInstances());

				Instance inst1 = unlabeled_view1.instance(index);
				u_prime1.add(inst1);
				unlabeled_view1.delete(index);

				Instance inst2 = unlabeled_view2.instance(index);
				u_prime2.add(inst2);
				unlabeled_view2.delete(index);
			}
		} // end of the iterations loop
				
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
		fw.close();
        
	}
	
	//cross - adding the most confident examples by both classifiers (no conflicting labels) 
	//with subsampling
	public static void testCoTrainingBlumMitchellNoRatio3(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int seed, int sample_size, int no_iter,
			String cType1, String cType2, 
			String output) throws Exception {
		
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		Random random;
		
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	   	
	  
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		
		//create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();
		
		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();
		
		//sampling u_prime1 and u_prime2
		random = new Random(seed);
		for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
			int index = random.nextInt(unlabeled_view1.numInstances());
			
			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);
			
			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} //done sampling u_prime1 and u_prime2
		
		//sum is the number of elements to be added to labeled data at each iteration	

		double[] class_distr_1;
		double[] class_distr_2;

		int crt_iter=0;
			
		while(crt_iter < no_iter && unlabeled_view1.numInstances()>0) {
			
			//crt_iter++;
			System.out.println("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances()+"\n\n");
			
			System.out.println("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances());
			fw.write("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances()+"\n\n");
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
			
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
			  				
			int numc1added=0;
			int numc2added=0;

			//classify u_prime1 with the classifier c1
			//if the probability is greater than, equal to the confidence-level 
			//add to the opposite set

			Hashtable<Integer, String> toadd_assigns1 = new Hashtable<Integer, String>();
			Hashtable<Integer, String> toadd_assigns2 = new Hashtable<Integer, String>();
			
			for(int j = 0; j < u_prime1.numInstances(); j ++) {

				Instance u_prime1_j = u_prime1.instance(j);							
				class_distr_1 = csc1.distributionForInstance(u_prime1_j);
				double predicted = Utils.maxIndex(class_distr_1);
				String value = u_prime1.classAttribute().value((int)predicted);
				
				double max=class_distr_1[(int)predicted];
				if (max >= confidence_high)
				{	
					toadd_assigns1.put(j, value);
				}
			}

			//classify u_prime2 with the classifier c2
			for(int j = 0; j < u_prime2.numInstances(); j++) {

				Instance u_prime2_j = u_prime2.instance(j);						
				class_distr_2 = csc2.distributionForInstance(u_prime2_j);
				double predicted = Utils.maxIndex(class_distr_2);
				String value = u_prime2.classAttribute().value((int)predicted);
				
				double max=class_distr_2[(int)predicted];				
				
				String value1 = toadd_assigns1.get(j);
				
				if (max >= confidence_high)
				{	
					if (value1==null)
					{
						toadd_assigns2.put(j, value);
					} else {
						if(!value1.equals(value))
							toadd_assigns1.remove(j);
						else
							toadd_assigns2.put(j, value);
					}
				}							
			}
			
			for(int j = 0; j < u_prime2.numInstances(); j++) {
				
				String value1 = toadd_assigns1.get(j);
				if (value1==null)
					continue;
		
				Instance temp = u_prime2.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view2.add(temp);
				numc2added++;					

			}
			
			for(int j = 0; j < u_prime1.numInstances(); j++) {
				
				String value1 = toadd_assigns2.get(j);
				if (value1==null)
					continue;
		
				Instance temp = u_prime1.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view1.add(temp);
				numc1added++;					

			}
			
			System.out.println("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added);
			fw.write("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added+"\n");
			
			u_prime1.delete();
			u_prime2.delete();
			
			crt_iter++;

			//sample from unlabeled for a new sample of U'
			random = new Random(seed);
			for(int n_u = 0; n_u < sample_size && unlabeled_view1.numInstances()>0; n_u++) {
				int index = random.nextInt(unlabeled_view1.numInstances());

				Instance inst1 = unlabeled_view1.instance(index);
				u_prime1.add(inst1);
				unlabeled_view1.delete(index);

				Instance inst2 = unlabeled_view2.instance(index);
				u_prime2.add(inst2);
				unlabeled_view2.delete(index);
			}
		} // end of the iterations loop
				
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
		fw.close();
        
	}
	
	//adding the most confident examples by both classifiers (no conflicting labels) 
	// no subsampling
	public static void testCoTrainingBlumMitchellNoRatio2_4(Instances train_view1, Instances valid_view1, 
			Instances train_view2, Instances valid_view2, 
			Instances unlabeled_view1, Instances unlabeled_view2, 
			Instances test_view1, Instances test_view2,
			int no_iter, String cType1, String cType2, 
			String output) throws Exception {
		
		PrintWriter fw = new PrintWriter(new FileWriter(output), true);
		
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;
		
		
	  	System.out.println("\nNumber of labeled instances: " + train_view1.numInstances());
	  	fw.write("Number of labeled instances: " + train_view1.numInstances() +  "\n\n");
	  	
	  	System.out.println("\nNumber of unlabeled instances: " + unlabeled_view1.numInstances());
	  	fw.write("Number of unlabeled instances: " + unlabeled_view1.numInstances() +  "\n\n");
	  	   	
	  
		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);
		

		double[] class_distr_1;
		double[] class_distr_2;

		int crt_iter=0;
			
		while(crt_iter < no_iter && unlabeled_view1.numInstances()>0) {
			
			//crt_iter++;
			System.out.println("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances());
			fw.write("Number of labeled instances in view 1: " + 
					labeled_view1.numInstances()+"\n\n");
			
			System.out.println("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances());
			fw.write("Number of labeled instances in view 2: " + 
					labeled_view2.numInstances()+"\n\n");
			
			Classifier basec1 = new SMO();
			Classifier basec2 = new SMO();
			if(cType1.equals("NBM")){
				basec1 = new NaiveBayesMultinomial();				
			} else if (cType1.equals("NB")){
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")){
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")){
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")){
				basec1 = new Bagging();
			} else if (cType1.equals("RF")){
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}
			
			if(cType2.equals("NBM")){
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")){
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")){
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")){
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")){
				basec2 = new Bagging();
			} else if (cType2.equals("RF")){
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}
			
			System.out.println("\nTrain on iteration: " + crt_iter);
/*			//learn classifiers from the set of labeled instances
			//c1 = new SMO();
			//((SMO)c1).setBuildLogisticModels(true);
			c1.buildClassifier(labeled_view1);
		
			//c2 = new SMO();
			//((SMO)c2).setBuildLogisticModels(true);
			c2.buildClassifier(labeled_view2);
*/			
			
			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);
			
			if (crt_iter==0)
			{
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Initial Predictions on the test set\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			} else {
				
				Evaluation eval_c1 = new Evaluation(test_view1);
			  	Evaluation eval_c2 = new Evaluation(test_view2);
			  	
			  	System.out.println("Test on iteration: " + crt_iter);
			  	fw.write("*********************************************\n");
				fw.write("Predictions on the test set\n");
				fw.write("Iteration: " + crt_iter + "\n");
		        eval_c1.evaluateModel(csc1, test_view1);
		        fw.write("Classifier 1\n");
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				eval_c2.evaluateModel(csc2, test_view2);
				fw.write("Classifier 2\n");
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("*********************************************\n");
			}
			
			if (valid_view1 != null && valid_view2 != null){
				
				Evaluation eval_c1 = new Evaluation(valid_view1);
			  	Evaluation eval_c2 = new Evaluation(valid_view2);
			  	
				System.out.println("Validation on iteration: " + crt_iter);
				fw.write("=============================================\n");
				fw.write("Predictions on the validation set\n");
				fw.write("Iteration: " + crt_iter + "\n");
				fw.write("Classifier 1\n");
		        eval_c1.evaluateModel(csc1, valid_view1);
		        fw.write(eval_c1.toClassDetailsString() + "\n");
				fw.write(eval_c1.toSummaryString() + "\n");
				fw.write(eval_c1.toMatrixString() + "\n");
				fw.write("Classifier 2\n");
				eval_c2.evaluateModel(csc2, valid_view2);
		        fw.write(eval_c2.toClassDetailsString() + "\n");
				fw.write(eval_c2.toSummaryString() + "\n");
				fw.write(eval_c2.toMatrixString() + "\n");
				fw.write("=============================================\n");
			}
			  				
			int numc1added=0;
			int numc2added=0;

			//classify u_prime1 with the classifier c1
			//if the probability is greater than, equal to the confidence-level 
			//add to the opposite set

			Hashtable<Integer, String> toadd_assigns = new Hashtable<Integer, String>();
			
			for(int j = 0; j < unlabeled_view1.numInstances(); j ++) {

				Instance unlabeled_view1_j = unlabeled_view1.instance(j);							
				class_distr_1 = csc1.distributionForInstance(unlabeled_view1_j);
				double predicted = Utils.maxIndex(class_distr_1);
				String value = unlabeled_view1.classAttribute().value((int)predicted);
				
				double max=class_distr_1[(int)predicted];
				if (max >= confidence_high)
				{	
					toadd_assigns.put(j, value);
				}
			}

			//classify u_prime2 with the classifier c2
			for(int j = 0; j < unlabeled_view2.numInstances(); j++) {

				Instance unlabeled_view2_j = unlabeled_view2.instance(j);						
				class_distr_2 = csc2.distributionForInstance(unlabeled_view2_j);
				double predicted = Utils.maxIndex(class_distr_2);
				String value = unlabeled_view2.classAttribute().value((int)predicted);
				
				double max=class_distr_2[(int)predicted];				
				
				String value1 = toadd_assigns.get(j);
				
				if (max >= confidence_high)
				{	
					if (value1==null)
					{
						toadd_assigns.put(j, value);
					} else {
						if(!value1.equals(value))
							toadd_assigns.remove(j);
					}
				}							
			}
			
			ArrayList<Integer> al = new ArrayList<Integer>();
			
			for(int j = 0; j < unlabeled_view2.numInstances(); j++) {
				
				String value1 = toadd_assigns.get(j);
				if (value1==null)
					continue;
				
				al.add(j);
		
				Instance temp = unlabeled_view1.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view1.add(temp);
				numc1added++;

				temp = unlabeled_view2.instance(j);
				temp.setClassMissing();
				temp.setClassValue(value1);
				labeled_view2.add(temp);
				numc2added++;					

			}
			
			int[] indices = new int[al.size()];
			for(int j = 0; j < al.size(); j++)
				indices[j] = al.get(j);
			Arrays.sort(indices,0,al.size());
			
			System.out.println("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added);
			fw.write("Iteration: " + crt_iter+" numc1added:"+numc1added+" numc2added:"+numc2added+"\n");
			
			for(int j = indices.length-1; j>0; j--){
				unlabeled_view1.delete(indices[j]);
				unlabeled_view2.delete(indices[j]);
			}
			
			
			crt_iter++;
			
		} // end of the iterations loop
				
		
		double[][] prod = new double[test_view1.numInstances()][];
		
		double[] distrT_1;
		
		for(int j = 0;j < test_view1.numInstances(); j++){
			
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			
			prod[j] = new double[distrT_1.length];
			for(int k = 0; k < distrT_1.length; k ++){
				prod[j][k] = distrT_1[k];
			}
		}
		
		double[] distrT_2;
		
		for(int j = 0;j < test_view2.numInstances(); j++){
			
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);
			
			for(int k = 0; k < distrT_2.length; k ++){
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				//prod[j][k] += distrT_2[k];
			}
		}
			
		for(int j = 0;j < prod.length; j++){
			
			if(!Utils.eq(Utils.sum(prod[j]), 0)){
				Utils.normalize(prod[j]);
			}
		}
		
		Evaluation eval_c1 = new Evaluation(test_view1);
	  	Evaluation eval_c2 = new Evaluation(test_view2);
	  	
	  	fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
        eval_c1.evaluateModel(csc1, test_view1);
        fw.write("Classifier 1\n");
        fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fw.write("Classifier 2\n");
        fw.write(eval_c2.toClassDetailsString() + "\n");
		fw.write(eval_c2.toSummaryString() + "\n");
		fw.write(eval_c2.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();
		
		for(int j = 0;j < prod.length; j++){
			
			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double)actual_idx,prod[j]);
			predictions.addElement(np);
		}
		
		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		
		ThresholdCurve ROC = new ThresholdCurve();
        Instances curve = ROC.getCurve(predictions, 0);
		
        fw.write("\n\nROC for combined classifier: \n\n");
        fw.write(curve.toString());
		
		fw.close();
        
	}
}
