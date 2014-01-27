package cotraining;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class Utilities {
	
	//check if the labels of two sets of instances are the same
	public static int differences(Instances p1, Instances p2){
		
		assert(p1.numInstances() == p2.numInstances());
		
		int n = 0;
		for(int i = 0; i < p1.numInstances(); i ++){
			Instance i1 = p1.instance(i);
			Instance i2 = p2.instance(i);
			
			String val1 = i1.stringValue(i1.classAttribute());
			String val2 = i2.stringValue(i2.classAttribute());
			
			if (!val1.equals(val2))
				n++;
		}
		return n;
	}


	public static void saveStringToFile(String output, String fileName){
		
		try {
			File outFile = new File(fileName);
			FileWriter outFileWriter = new FileWriter(outFile);
			outFileWriter.write(output);
			outFileWriter.close();	
		}
		catch (Exception e){
			System.out.println("Cannot write file...");
		}	
	}
	
	
    public int numWithClass(Instances data, String classLabel){
        int count = 0;
        Attribute classAttr = data.classAttribute();
        
        for (int i = 0; i < data.numInstances(); i ++){
            Instance current = data.instance(i);
            if ( (current.stringValue(classAttr)).equals(classLabel))
                count ++;
        }
        
        return count;
    }
    
	public Instances getSubset(Instances data, int seed, double sizeF){
		
		Random R = new Random(seed);
		
		//extracting class values
		Attribute classAttr = data.classAttribute();
		String [] cValues = new String[classAttr.numValues()];
		
		for (int i = 0; i < cValues.length; i ++){
			cValues[i] = classAttr.value(i);
		}
		
		//separating data by class
		Instances [] dataByClass = new Instances[cValues.length];
		
		for (int i = 0; i < dataByClass.length; i ++){
			dataByClass[i] = new Instances(data);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < data.numInstances(); i ++){
			Instance current = data.instance(i);
			
			for (int j = 0; j < dataByClass.length; j ++){
				if (current.stringValue(classAttr).equals(cValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		double[] ratio = new double[cValues.length];
		
		//computing ratio
		for (int j = 0; j < ratio.length; j ++){
				ratio[j] = (double)dataByClass[j].numInstances() / (double)data.numInstances();
		}
		
		//computing ratio CDF
		double [] ratioCDF = new double[ratio.length];
		for (int i = 0; i < ratio.length; i ++){
			if (i == 0)
				ratioCDF[i] = ratio[i];
			else 
				ratioCDF[i] = ratio[i]+ratioCDF[i-1];
		}
		
		//creating subset
		Instances subsetData = new Instances(data);
		subsetData.delete();
		
		int subsetSize = (int)Math.round(sizeF*(double)data.numInstances());
		
		for (int i = 0; i < subsetSize; i ++){
			double rand = R.nextDouble();
			
			for (int j = 0; j < ratioCDF.length; j ++){
				if (rand <= ratioCDF[j]){
					int randomInt = (int)Math.floor(R.nextDouble()*(double)dataByClass[j].numInstances());
					Instance instanceToAdd = dataByClass[j].instance(randomInt);
					subsetData.add(instanceToAdd);
					dataByClass[j].delete(randomInt);
					j = ratioCDF.length;
					break;
				}
			}
		}
		
		double[] ratioF = new double[cValues.length];
		
		int[] num = new int[cValues.length];
		
		//computing ratio
		for (int j = 0; j < ratioF.length; j ++){
			num[j] = this.numWithClass(subsetData, cValues[j]);
				ratioF[j] = (double)num[j] / (double)subsetData.numInstances();
		}
		
		return subsetData;
	}
	
	
	
	
	public Instances getFractionUnlab(Instances unlabeled_data, int seed, double sizeF){
		
		Random R = new Random(seed);	
		Instances m_unlabeled_data = new Instances (unlabeled_data);
		
		//creating subset
		Instances subsetData = new Instances(unlabeled_data);
		subsetData.delete();
		
		int subsetSize = (int)Math.round(sizeF*(double)m_unlabeled_data.numInstances());
		
		for (int i = 0; i < subsetSize; i ++){
			
			int index = R.nextInt(m_unlabeled_data.numInstances());
			Instance inst = m_unlabeled_data.instance(index);
			subsetData.add(inst);
			m_unlabeled_data.delete(index);
		}
		
		return subsetData;
	}

	public Instances getFractionLab(Instances labeled_data, int seed, double sizeF_l){
		
		Random R = new Random(seed);	
		Instances m_labeled_data = new Instances (labeled_data);
		
		//creating subset
		Instances subsetData = new Instances(labeled_data);
		subsetData.delete();
	
		//get class values
		Attribute classAttr = labeled_data.classAttribute();
		String [] cValues = new String[classAttr.numValues()];
		
		for (int i = 0; i < cValues.length; i ++){
			cValues[i] = classAttr.value(i);
		}
		
		//separate data by class
		Instances [] dataByClass = new Instances[cValues.length];
		
		for (int i = 0; i < dataByClass.length; i ++){
			dataByClass[i] = new Instances(labeled_data);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < labeled_data.numInstances(); i ++){
			Instance current = labeled_data.instance(i);
			
			for (int j = 0; j < dataByClass.length; j ++){
				if (current.stringValue(classAttr).equals(cValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		//compute distribution in data
		double[] ratio = new double[cValues.length];
		
		for (int j = 0; j < ratio.length; j ++){
				ratio[j] = (double)dataByClass[j].numInstances() / (double)labeled_data.numInstances();
		}
						
		int subsetSize_L = (int)Math.round(sizeF_l*(double)labeled_data.numInstances());
			
		int[] L_byClass = new int[cValues.length];
		int l = 0;
		
		for (int j = 0; j < L_byClass.length-1; j ++){
			L_byClass[j] = (int)Math.round(ratio[j]*(double)subsetSize_L);
			l += L_byClass[j];
		}
		
		L_byClass[cValues.length-1] = subsetSize_L - l;
			
		for(int j = 0; j < L_byClass.length; j ++) {		
				for(int i = 0; i < L_byClass[j];i ++) {
					int index = R.nextInt(m_labeled_data.numInstances());
					Instance instance = dataByClass[j].instance(index);
					dataByClass[j].delete(index);
					subsetData.add(instance);
				}
		}
			
		return subsetData;
	}


    
	public Instances getBalancedSubsetAllPositive(Instances trainData, String posLabel,
			Random R){
		
		//separate dataset into positive and negative
		Instances posInstances = new Instances(trainData);
		posInstances.delete();
		
		Instances negInstances = new Instances(trainData);
		negInstances.delete();
		
		Attribute classAttr = trainData.classAttribute();
		
		for (int i = 0; i < trainData.numInstances(); i ++){
			
            Instance current = trainData.instance(i);
			if (current.stringValue(classAttr).equals(posLabel)) {
				posInstances.add(current);
			}
			else {
				negInstances.add(current);
            }
		}
		//System.out.println("pos: " + posInstances.numInstances());
		//System.out.println("neg: " + negInstances.numInstances());
		
		Instances subset = new Instances(trainData);
		subset.delete();
		
		//adding all positives
		for (int i = 0; i < posInstances.numInstances(); i ++){
			Instance current = posInstances.instance(i);
			subset.add(current);
		}
		
		//adding the equal number of negative instances
		negInstances.randomize(R);
		
		int max = 0;
	
		if (posInstances.numInstances() <= negInstances.numInstances())
		{
			max = posInstances.numInstances();
		}
		else
		{
			max = negInstances.numInstances();
		}
		
		for (int i = 0; i < max; i ++){
			
			Instance current;
			
			//sample a negative instances
			current = negInstances.instance(i);
			subset.add(current);
		}
		return subset;
	}
	
	public static Instances[] getFractions(Instances train, double sizeF_l, double sizeF_u, int seed){
		
		Instances[] subsets = new Instances[2];
		//the labeled subset
		subsets[0] = new Instances(train);
		subsets[0].delete();
		//the unlabeled subset
		subsets[1] = new Instances(train);
		subsets[1].delete();
		
		//get class values
		Attribute clsAttr = train.classAttribute();
		String[] clsValues = new String[clsAttr.numValues()];
		
		for(int i = 0; i < clsValues.length; i++){
			clsValues[i] = clsAttr.value(i);
		}
		
		//separate data by class
		Instances[] dataByClass = new Instances[clsValues.length];
		
		for(int i = 0; i < dataByClass.length; i++){
			dataByClass[i] = new Instances(train);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < train.numInstances(); i ++){
			Instance current = train.instance(i);
			
			for(int j = 0; j < clsValues.length; j++){
				if(current.stringValue(clsAttr).equals(clsValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		//randomize instances
		Random random = new Random(seed);
		for(int i = 0; i < dataByClass.length; i++){
			dataByClass[i].randomize(random);
		}
		
		//compute distribution in data
		double[] ratio = new double[clsValues.length];
		
		for (int j = 0; j < ratio.length; j ++){
				ratio[j] = (double)dataByClass[j].numInstances() / (double)train.numInstances();
		}
		
		//determine the number of instances from each class in unlabeled data
		int subsetSize_U = (int)Math.round(sizeF_u*(double)train.numInstances());
		
		int[] U_byClass = new int[clsValues.length];
		int u = 0;
		for (int j = 0; j < U_byClass.length-1; j ++){
			U_byClass[j] = (int)Math.round(ratio[j]*(double)subsetSize_U);
			u += U_byClass[j];
		}
		U_byClass[clsValues.length-1] = subsetSize_U - u;
		
		//create the unlabeled set
		for(int j = 0; j < U_byClass.length; j ++) {
			
			for(int i = dataByClass[j].numInstances()-1; i > dataByClass[j].numInstances()-1-U_byClass[j];i --) {
				Instance instance = dataByClass[j].instance(i);
				subsets[1].add(instance);
			}
		}
		
		double size = sizeF_u + sizeF_l;
		
		if (size == 1.0) {
			
			//the rest of train data goes to the labeled set
			for (int j = 0; j < U_byClass.length; j ++){
				
				for (int i = 0; i < dataByClass[j].numInstances()-U_byClass[j]; i ++){
					Instance instance = dataByClass[j].instance(i);
					subsets[0].add(instance);
				}
			}
		} else if ((size < 1.0) && (size > 0.0)) {
			
			//determine the number of instances from each class in labeled data
			int subsetSize_L = (int)Math.round(sizeF_l*(double)train.numInstances());
			
			int[] L_byClass = new int[clsValues.length];
			int l = 0;
			for (int j = 0; j < L_byClass.length-1; j ++){
				L_byClass[j] = (int)Math.round(ratio[j]*(double)subsetSize_L);
				int sum = L_byClass[j] + U_byClass[j];
				if(sum > dataByClass[j].numInstances()){
					L_byClass[j] = dataByClass[j].numInstances() - U_byClass[j];
				}
				l += L_byClass[j];
			}
			L_byClass[clsValues.length-1] = subsetSize_L - l;
			
			//create the labeled set
			for(int j = 0; j < L_byClass.length; j ++) {
				
				for(int i = 0; i < L_byClass[j];i ++) {
					Instance instance = dataByClass[j].instance(i);
					subsets[0].add(instance);
				}
			}
		} else {
			System.out.println("size must be between 0 and 1");
		}
		
		return subsets;
	}
	
	public static Instances getBalanced(Instances train, int seed){
		
		//get class values
		Attribute clsAttr = train.classAttribute();
		String[] clsValues = new String[clsAttr.numValues()];
		
		for(int i = 0; i < clsValues.length; i++){
			clsValues[i] = clsAttr.value(i);
		}

		//separate data by class
		Instances[] dataByClass = new Instances[clsValues.length];
		
		for(int i = 0; i < dataByClass.length; i++){
			dataByClass[i] = new Instances(train);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < train.numInstances(); i ++){
			Instance current = train.instance(i);
			
			for(int j = 0; j < clsValues.length; j++){
				if(current.stringValue(clsAttr).equals(clsValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		Random random = new Random(seed);
		
		Instances subset = new Instances(train);
		subset.delete();
		
		//randomize instances
		for(int i = 0; i < dataByClass.length; i++){
			dataByClass[i].randomize(random);
		}
		
		//determine the class with smaller number of instances
		int n = 0;
		if(dataByClass[0].numInstances()<=dataByClass[1].numInstances())
			n = dataByClass[0].numInstances();
		else n = dataByClass[1].numInstances();
		
		//create the balanced set
		for (int i = 0; i < n; i ++){
			
			subset.add(dataByClass[0].instance(i));
			subset.add(dataByClass[1].instance(i));
		}
		return subset;
	}
	
	public Instances[] getUprimes(int sample_size, int seed, Instances unlabeled_view_1, Instances unlabeled_view_2)
	{
		Instances[] Uprimes = new Instances[2];
		
		Random random = new Random(seed);
		
		//clean U' in view 1
		Instances u_prime_1 = new Instances(unlabeled_view_1);
		u_prime_1.delete();
		
		//clean U' in view 2
		Instances u_prime_2 = new Instances(unlabeled_view_2);
		u_prime_2.delete();
		
		//random sample instanced from unlabeled and add them to U'
		for(int n_u = 0; n_u < sample_size; n_u++){
				
				int index = random.nextInt(unlabeled_view_1.numInstances());
				
				Instance inst_1 = unlabeled_view_1.instance(index);
				u_prime_1.add(inst_1);
				unlabeled_view_1.delete(index);
				
				Instance inst_2 = unlabeled_view_2.instance(index);
				u_prime_2.add(inst_2);
				unlabeled_view_2.delete(index);
		}
		
		Uprimes[0] = u_prime_1;
		Uprimes[1] = u_prime_2;
		
		return Uprimes;

	}
	
	public Instances[] doubleUprimes(int sample_size, int seed, Instances u_prime_1, Instances unlabeled_view_1, Instances u_prime_2, Instances unlabeled_view_2)
	{
		Instances[] Uprimes = new Instances[2];
		
		Random random = new Random(seed);
			
		//random sample instanced from unlabeled and add them to U'
		for(int n_u = 0; n_u < sample_size; n_u++){
				
				int index = random.nextInt(unlabeled_view_1.numInstances());
				
				Instance inst_1 = unlabeled_view_1.instance(index);
				u_prime_1.add(inst_1);
				unlabeled_view_1.delete(index);
				
				Instance inst_2 = unlabeled_view_2.instance(index);
				u_prime_2.add(inst_2);
				unlabeled_view_2.delete(index);
		}
		
		Uprimes[0] = u_prime_1;
		Uprimes[1] = u_prime_2;
		
		return Uprimes;

	}
	
	public Instances[] getFractions(Instances data, double sizeF_l, double sizeF_u){
		
		Instances[] subsets = new Instances[2];
		subsets[0] = new Instances(data);
		subsets[0].delete();
		subsets[1] = new Instances(data);
		subsets[1].delete();
		
		//get class values
		Attribute classAttr = data.classAttribute();
		String [] cValues = new String[classAttr.numValues()];
		
		for (int i = 0; i < cValues.length; i ++){
			cValues[i] = classAttr.value(i);
		}
		
		//separate data by class
		Instances [] dataByClass = new Instances[cValues.length];
		
		for (int i = 0; i < dataByClass.length; i ++){
			dataByClass[i] = new Instances(data);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < data.numInstances(); i ++){
			Instance current = data.instance(i);
			
			for (int j = 0; j < dataByClass.length; j ++){
				if (current.stringValue(classAttr).equals(cValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		//compute distribution in data
		double[] ratio = new double[cValues.length];
		
		for (int j = 0; j < ratio.length; j ++){
				ratio[j] = (double)dataByClass[j].numInstances() / (double)data.numInstances();
		}
		
		int subsetSize_U = (int)Math.round(sizeF_u*(double)data.numInstances());
		
		int[] U_byClass = new int[cValues.length];
		int u = 0;
		for (int j = 0; j < U_byClass.length-1; j ++){
			U_byClass[j] = (int)Math.round(ratio[j]*(double)subsetSize_U);
			u += U_byClass[j];
		}
		U_byClass[cValues.length-1] = subsetSize_U - u;
		
		for(int j = 0; j < U_byClass.length; j ++) {
			
			for(int i = dataByClass[j].numInstances()-1; i > dataByClass[j].numInstances()-1-U_byClass[j];i --) {
				Instance instance = dataByClass[j].instance(i);
				subsets[1].add(instance);
			}
		}
		
		double size = sizeF_u + sizeF_l;
		
		if (size == 1.0) {
			
			for (int j = 0; j < U_byClass.length; j ++){
				
				for (int i = 0; i < dataByClass[j].numInstances()-U_byClass[j]; i ++){
					Instance instance = dataByClass[j].instance(i);
					subsets[0].add(instance);
				}
			}
		} else {
			
			int subsetSize_L = (int)Math.round(sizeF_l*(double)data.numInstances());
			
			int[] L_byClass = new int[cValues.length];
			int l = 0;
			for (int j = 0; j < L_byClass.length-1; j ++){
				L_byClass[j] = (int)Math.round(ratio[j]*(double)subsetSize_L);
				int sum = L_byClass[j] + U_byClass[j];
				if(sum > dataByClass[j].numInstances()){
					L_byClass[j] = dataByClass[j].numInstances() - U_byClass[j];
				}
				l += L_byClass[j];
			}
			L_byClass[cValues.length-1] = subsetSize_L - l;
			
			for(int j = 0; j < L_byClass.length; j ++) {
				
				for(int i = 0; i < L_byClass[j];i ++) {
					Instance instance = dataByClass[j].instance(i);
					subsets[0].add(instance);
				}
			}
		}
		
		return subsets;
	}
	
	public Instances getFractionLabeled(Instances data, double sizeF_l)
	{
		Instances[] subsets = new Instances[2];
		subsets[0] = new Instances(data);
		subsets[0].delete();
		subsets[1] = new Instances(data);
		subsets[1].delete();
		subsets = getFractions(data, sizeF_l, 1-sizeF_l);
		return subsets[0];
	}
	
	public static Instances getFractionLabeled(Instances data, double sizeF_l, int seed)
	{
		Instances[] subsets = new Instances[2];
		subsets[0] = new Instances(data);
		subsets[0].delete();
		subsets[1] = new Instances(data);
		subsets[1].delete();
		subsets = getFractions(data, sizeF_l, 0.0, seed);
		return subsets[0];
	}
	
	public static int[] getExpRatios(Instances data){
		
		//get class values
		Attribute classAttr = data.classAttribute();
		String [] cValues = new String[classAttr.numValues()];
		
		for (int i = 0; i < cValues.length; i ++){
			cValues[i] = classAttr.value(i);
		}
		
		//separate data by class
		Instances [] dataByClass = new Instances[cValues.length];
		
		for (int i = 0; i < dataByClass.length; i ++){
			dataByClass[i] = new Instances(data);
			dataByClass[i].delete();
		}
		
		for (int i = 0; i < data.numInstances(); i ++){
			Instance current = data.instance(i);
			
			for (int j = 0; j < dataByClass.length; j ++){
				if (current.stringValue(classAttr).equals(cValues[j])){
					dataByClass[j].add(current);
				}
			}
		}
		
		int[] noByClass = new int[cValues.length];
		for (int j = 0; j < noByClass.length; j ++){
			noByClass[j] = dataByClass[j].numInstances();
		}
		int min = Utils.minIndex(noByClass);
		double val = noByClass[min];
		
		for (int j = 0; j < noByClass.length; j ++){
			noByClass[j] = (int) Math.floor((double)noByClass[j] / val);
		}
		
		return noByClass;
	}
	
	
	public static void main(String[] args) throws Exception {
		
		String fileName = "/Users/cornelia/Desktop/workspaceHACL_icml/AbstractionMMk/src/data/data.arff";
		
		DataSource source = null;
    	source = new DataSource(fileName);
    	Instances instances = source.getDataSet();
    	instances.setClassIndex(instances.numAttributes()-1);
    	
    	Utilities u = new Utilities();
    	u.getSubset(instances, 0, 1);
	}
	
	
}
