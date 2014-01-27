package cotraining;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.sql.Date;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

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
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class CoTrainingExperiments {

	private static final String path = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/september9/10000/arff_facial/";

	private static final int trainSetPartitionStart = 0;
	private static final int trainSetPartitionSize = 2000;

	private static final int unlabeledSetPartitionStart = 2000;
	private static final int unlabeledSetPartitionSize = 2500;

	private static final int testSetPartitionStart = 4200;
	private static final int testSetPartitionSize = 500;

	private static  int randomSeed = 100;// 100,200,250

	private static HashMap<Integer, String> map ;
	private static List<Integer> chosenIndices = new ArrayList<Integer>();
	private static String getResultLocation(String relationName) {
		return "results/" + relationName + "/randomSeed_" + randomSeed + "/";
	}

	public static void main (String[] args)  throws Exception
	{
		map = getURLMap(new File(path+"10000_unique_metadata"));
//		randomSeed = 100;
//		main1(args);
//		randomSeed = 300;
//		main1(args);
		randomSeed = 200;                                           
		main1(args);
	}
	
	public static void main1(String[] args) throws Exception {
		CoTrainingExperiments exp = new CoTrainingExperiments();
		
		// exp.baseLineLowRun(1400, path);
		// exp.baseLineHighRun(1400, path);
		// exp.testCoTrainginRun(1500, path);
		// exp.baseLineLowCombinedRun(1400, path);
		// exp.testCoTrainginRun(1400, path);
		// exp.testCoTrainginRun(1400, path);
		
		// exp.baseLineLowSingleClassifier(path);
		//exp.baseLineHighRun(2000, path);
//		exp.baseLineHighCombinedRun(2000, path);
		for (int i = 100; i < 1300;) {
			// exp.baseLineLowRun(i, path);
			// exp.testCoTrainginRun(i, path);
			exp.baseLineHighCombinedRun(i, path);
			//exp.baseLineLowSingleClassifier( path, i);
			//exp.baseLineLowSingleClassifier( path, i);
			i = i + 200;
		}
	}

	public static HashMap<Integer,String> getURLMap(File metaFile) {
		HashMap<Integer, String> map = new HashMap<Integer, String>();
		if (!metaFile.exists())
			return map;
		FileInputStream inputStream;
		try {
			inputStream = new FileInputStream(metaFile);
			DataInputStream in = new DataInputStream(inputStream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String url;
			while ((url = br.readLine()) != null) {
				String[] urlLines = url.split("jpg:");
				String urlLine = urlLines[0];
				String[] ulrkp = urlLine.split("-");
				int key = Integer.parseInt( ulrkp[0] );
				String val = ulrkp[1] + "jpg";
				map.put(key, val.trim());
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return map;
	}

	public static List<String> getURLList(File metaFile) {
		List<String> urlList = new ArrayList<String>();
		if (!metaFile.exists())
			return urlList;
		FileInputStream inputStream;
		try {
			inputStream = new FileInputStream(metaFile);
			DataInputStream in = new DataInputStream(inputStream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String url;
			while ((url = br.readLine()) != null) {
				String[] urlLines = url.split("jpg:");
				String urlLine = urlLines[0];
				String[] ulrkp = urlLine.split("-");
				int key = Integer.parseInt( ulrkp[0] );
				String val = ulrkp[1] + "jpg";
				urlList.add(val.trim());
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return urlList;
	}
	
	
	public ClassInstances createClassInstances(String classifierPath,
			int trainSize, int unlabeledSize, int testSize) throws Exception {
		DataSource source = new DataSource(classifierPath);
		Instances classSet1 = source.getDataSet();
		// ////
		// String result_path = getResultLocation("fullclassSet");
		// String op = classSet1.relationName() ;
		// File dir = new File(path+result_path);
		// boolean dirCreated = dir.mkdirs();
		// if(!dirCreated)
		// System.err.println("error creating dir "+ path+result_path);
		// PrintWriter fw = null;
		// try {
		// fw = new PrintWriter(new FileWriter(path+result_path+op), true);
		// } catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		// fw.write(classSet1.toString());
		// ///
		Instances trainSet1 = getInstances(classSet1, trainSetPartitionStart,
				trainSize, trainSetPartitionSize, new Random(randomSeed));
		Instances unlabeledSet1 = getInstances(classSet1,
				unlabeledSetPartitionStart, unlabeledSize,
				unlabeledSetPartitionSize, new Random(randomSeed));
		Instances testSet1 = getInstances(classSet1, testSetPartitionStart,
				testSize, testSetPartitionSize, new Random(randomSeed));

		printToFile(trainSet1, "train");
		printToFile(testSet1, "test");
		ClassInstances cl = new ClassInstances(trainSet1, unlabeledSet1, testSet1);
		return cl;
	}

	private void printToFile(Instances trainSet1, String setType) {
		String relation = trainSet1.relationName();
		String result_path = getResultLocation(trainSet1.relationName());
		System.out.println(path + result_path + setType + "/");
		File dir = new File(path + result_path + setType + "/");
		boolean dirCreated = dir.mkdirs();
		if (!dirCreated)
			System.err.println("error creating dir " + path + result_path);
		try {
			PrintWriter fw = new PrintWriter(new FileWriter(path + result_path
					+ setType + "/" + relation + ".arff"), true);
			fw.write(trainSet1.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private Instances getInstances(Instances classSet1, int start, int end,
			int randomSize, Random seed) {

		chosenIndices.clear();
		if (seed == null) {

			// System.out.println(classSet1.numInstances());
			Instances newSet = new Instances(classSet1, start, end);
			// System.out.println(classSet1.numInstances());
			newSet.setClassIndex(classSet1.numAttributes() - 1);
			return newSet;

		} else {
			Instances newSet = new Instances(classSet1, end);
			int setSize = end;
			for (int i = 0; i < setSize; i++) {
				int index = seed.nextInt(randomSize) + start;
				chosenIndices.add(index);
				// System.out.print(index +" ");
				newSet.add(classSet1.instance(index));
			}
			newSet.setClassIndex(classSet1.numAttributes() - 1);
			// System.out.println(newSet.numInstances());
			return newSet;
		}

	}

	public static Instances mergeInstances(Instances first, Instances second) {

		if (first.numInstances() != second.numInstances()) {
			throw new IllegalArgumentException(
					"Instance sets must be of the same size");
		}

		// Create the vector of merged attributes
		FastVector newAttributes = new FastVector();
		for (int i = 0; i < first.numAttributes() - 1; i++) {
			newAttributes.addElement(first.attribute(i));
		}
		for (int i = 0; i < second.numAttributes(); i++) {
			newAttributes.addElement(second.attribute(i));
		}

		// Create the set of Instances
		Instances merged = new Instances(first.relationName() + '_'
				+ second.relationName(), newAttributes, first.numInstances());
		// Merge each instance
		for (int i = 0; i < first.numInstances(); i++) {
			merged.add(mergeInstance(second.instance(i), first.instance(i)));
		}
		return merged;
	}

	private static Instance mergeInstance(Instance instance, Instance inst) {
		int m = 0;
		double[] newVals = new double[instance.numAttributes()
				+ inst.numAttributes() - 1];
		for (int j = 0; j < instance.numAttributes() - 1; j++, m++) {
			newVals[m] = instance.value(j);
		}
		for (int j = 0; j < inst.numAttributes(); j++, m++) {
			newVals[m] = inst.value(j);
		}
		return new SparseInstance(1.0, newVals);
	}

	public ClassInstances mergeInstances(ClassInstances classfierInstances1,
			ClassInstances classfierInstances2) {

		// String result_path = getResultLocation("mergedSet");
		// String op = classfierInstances1.getTestInstances().relationName() +
		// classfierInstances2.getTestInstances().relationName();
		// File dir = new File(path+result_path);
		// boolean dirCreated = dir.mkdirs();
		// if(!dirCreated)
		// System.err.println("error creating dir "+ path+result_path);
		// PrintWriter fw = null;
		// try {
		// fw = new PrintWriter(new FileWriter(path+result_path+op), true);
		// } catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		//
		ClassInstances mergedInstances = new ClassInstances();
		Instances mergedTest = mergeInstances(
				classfierInstances1.getTestInstances(),
				classfierInstances2.getTestInstances());
		mergedTest.setClassIndex(mergedTest.numAttributes() - 1);
		// for(int i = 0 ; i < mergedTest.numInstances() ;i++ )
		// {
		// System.out.println(mergedTest.instance(i).classValue());
		// System.out.println(mergedTest.instance(i).stringValue(mergedTest.numAttributes()
		// - 1));
		// System.out.println(mergedTest.instance(i));
		// }
		// System.out.println("cls1 -" +
		// classfierInstances1.getTestInstances().instance(0));
		// System.out.println("cls2 -" +
		// classfierInstances2.getTestInstances().instance(0));
		// System.out.println("merged -" + mergedTest.instance(0));

		Instances mergedUnlabeled = mergeInstances(
				classfierInstances1.getUnlabeledInstances(),
				classfierInstances2.getUnlabeledInstances());
		mergedUnlabeled.setClassIndex(mergedUnlabeled.numAttributes() - 1);

		Instances mergedTrain = mergeInstances(
				classfierInstances1.getTrainInstances(),
				classfierInstances2.getTrainInstances());
		mergedTrain.setClassIndex(mergedTrain.numAttributes() - 1);

		
		mergedInstances.setTestInstances(mergedTest);
		mergedInstances.setUnlabeledInstances(mergedUnlabeled);
		mergedInstances.setTrainInstances(mergedTrain);

		printToFile(mergedTrain, "train");
		printToFile(mergedTest, "test");

		// fw.close();

		return mergedInstances;
	}

	public void baseLineLowSingleClassifier(String path, int size) throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		String cType1 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		String classifierPath4 = "/classifier_facial.arff";
		
		String classifierPath5 = "/classifier_edge.arff";
		// loading classifier instances//

		int train_size = size;
		int unlabeled_size = 2000;
		int testSize = 500;
//		ClassInstances classfierInstances_Sift = this.createClassInstances(path
//				+ classifierPath1, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_Meta = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, testSize);
//		ClassInstances classfierInstances_Rgb = this.createClassInstances(path
//				+ classifierPath3, train_size, unlabeled_size, testSize);
//		ClassInstances classfierInstances_facial = this.createClassInstances(
//				path + classifierPath4, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_edge = this.createClassInstances(
				path + classifierPath5, train_size, unlabeled_size, testSize);
		// //

//		Instances trainSet_Sift = addInstanceSets(
//				classfierInstances_Sift.getTrainInstances(),
//				classfierInstances_Sift.getUnlabeledInstances());
		Instances trainSet_Meta = addInstanceSets(
				classfierInstances_Meta.getTrainInstances(),
				classfierInstances_Meta.getUnlabeledInstances());
//		Instances trainSet_Rgb = addInstanceSets(
//				classfierInstances_Rgb.getTrainInstances(),
//				classfierInstances_Rgb.getUnlabeledInstances());
//		Instances trainSet_Facial = addInstanceSets(
//				classfierInstances_facial.getTrainInstances(),
//				classfierInstances_facial.getUnlabeledInstances());
		Instances trainSet_Edge = addInstanceSets(
				classfierInstances_edge.getTrainInstances(),
				classfierInstances_edge.getUnlabeledInstances());
		// //

		double sizeF_l = 0.1;
		double sizeF_u = 0.5;
		String output = "baseLineLow_SingleClassifier:trainingSize_"
				+ train_size + "_" + "testSize" + testSize;
//		baseline_low_Single(classfierInstances_Sift.getTrainInstances(),
//				classfierInstances_Sift.getTestInstances(), cType1, sizeF_l,
//				output, path);
//
//		baseline_low_Single(trainSet_Meta,
//				classfierInstances_Meta.getTestInstances(), cType1, sizeF_l,
//				output, path);
//
//		baseline_low_Single(trainSet_Rgb,
//				classfierInstances_Rgb.getTestInstances(), cType1, sizeF_l,
//				output, path);
//
//
//		baseline_low_Single(trainSet_Facial,
//				classfierInstances_facial.getTestInstances(), cType1, sizeF_l,
//				output, path);
		
		baseline_low_Single(classfierInstances_edge.getTrainInstances(),
				classfierInstances_edge.getTestInstances(), cType1, sizeF_l,
				output, path);
		
		System.out.println("Done!\n");
	}

	public void baseLineLowCombinedRun(int trainsize, String path)
			throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		String cType1 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		String classifierPath4 = "/classifier_facial.arff";
		
		String classifierPath5 = "/classifier_edge.arff";

		// loading classifier instances//

		int train_size = trainsize;
		int unlabeled_size = 2000;
		String output = path + "testCoTraining_" + "trainingSize_" + train_size
				+ "_" + "unlabeledSIze" + "_" + unlabeled_size;
		ClassInstances classfierInstances1 = this.createClassInstances(path
				+ classifierPath1, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances2 = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances3 = this.createClassInstances(path
				+ classifierPath3, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances4 = this.createClassInstances(path
				+ classifierPath4, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances5 = this.createClassInstances(path
				+ classifierPath5, train_size, unlabeled_size, 500);
		// //

		ClassInstances mergedInstances = mergeInstances(classfierInstances3,
				classfierInstances5);
//		 ClassInstances allMerged = mergeInstances(mergedInstances,
//		 classfierInstances3);
		// ClassInstances allMerged2 = mergeInstances(allMerged,
		// classfierInstances4);
		double sizeF_l = 0.1;
		double sizeF_u = 0.5;
		output = "baseLineLow_combined" + "trainingSize_" + train_size + "_"
				+ "unlabeledSize_" + unlabeled_size;
		baseline_low_Single(mergedInstances.getTrainInstances(),
				mergedInstances.getTestInstances(), cType1, sizeF_l, output,
				path);

		System.out.println("Done!\n");
	}

	public void baseLineLowRun(int trainsize, String path) throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		String cType1 = new String("SVM");
		String cType2 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		// loading classifier instances//

		int train_size = trainsize;
		int unlabeled_size = 1600;
		String output = path + "testCoTraining_" + "trainingSize_" + train_size
				+ "_" + "unlabeledSIze" + "_" + unlabeled_size;
		ClassInstances classfierInstances_Sift = this.createClassInstances(path
				+ classifierPath1, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances_Meta = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, 500);
		ClassInstances classfierInstances_Rgb = this.createClassInstances(path
				+ classifierPath3, train_size, unlabeled_size, 500);

		// //

		ClassInstances mergedInstances = mergeInstances(
				classfierInstances_Sift, classfierInstances_Rgb);

		// ClassInstances mergedInstances2 =
		// mergeInstances(classfierInstances_Sift,
		// classfierInstances_Meta);

		double sizeF_l = 0.1;
		double sizeF_u = 0.5;
		output = "baseLineLow_" + "trainingSize_" + train_size + "_"
				+ "unlabeledSize_" + unlabeled_size;
		baseline_low(classfierInstances_Sift.getTrainInstances(),
				classfierInstances_Sift.getTestInstances(),
				classfierInstances_Meta.getTrainInstances(),
				classfierInstances_Meta.getTestInstances(), cType1, cType2,
				sizeF_l, output, path);

		System.out.println("Done!\n");
	}

	public void baseLineHighRun(int trainsize, String path) throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		String cType1 = new String("SVM");
		String cType2 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		String classifierPath4 = "/classifier_facial.arff";
		
		String classifierPath5 = "/classifier_edge.arff";

		// loading classifier instances//

		int train_size = trainsize;
		int unlabeled_size = 2300;
		int testSize = 500;
		String output = "baseLineHigh:" + "trainingSize_" + train_size + "_"
				+ "unlabeledSIze" + "_" + unlabeled_size;
		ClassInstances classfierInstances_Sift = this.createClassInstances(path
				+ classifierPath1, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_Meta = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_Rgb = this.createClassInstances(path
				+ classifierPath3, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_facial = this.createClassInstances(
				path + classifierPath4, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_edge = this.createClassInstances(
				path + classifierPath5, train_size, unlabeled_size, testSize);
		// //
		ClassInstances mergedIsntances = mergeInstances(
				classfierInstances_edge, classfierInstances_Sift);

//		ClassInstances facial_rb_facial = mergeInstances(mergedIsntances,
//				classfierInstances_Sift);

		Instances trainSet_Sift = addInstanceSets(
				classfierInstances_Sift.getTrainInstances(),
				classfierInstances_Sift.getUnlabeledInstances());
		Instances trainSet_Meta = addInstanceSets(
				classfierInstances_Meta.getTrainInstances(),
				classfierInstances_Meta.getUnlabeledInstances());
		Instances trainSet_Rgb = addInstanceSets(
				classfierInstances_Rgb.getTrainInstances(),
				classfierInstances_Rgb.getUnlabeledInstances());
		Instances trainSet_Facial = addInstanceSets(
				classfierInstances_facial.getTrainInstances(),
				classfierInstances_facial.getUnlabeledInstances());
		
		Instances trainSet_Edge = addInstanceSets(
				classfierInstances_edge.getTrainInstances(),
				classfierInstances_edge.getUnlabeledInstances());

		Instances mergedHigh = addInstanceSets(
				mergedIsntances.getTrainInstances(),
				mergedIsntances.getUnlabeledInstances());


		double sizeF_l = 0.1;
		double sizeF_u = 0.5;

		baseline_low(mergedHigh, 				mergedIsntances.getTestInstances(),
				trainSet_Meta, classfierInstances_Meta.getTestInstances(),
				cType1, cType2, sizeF_l, output, path);

		System.out.println("Done!\n");
	}

	public void baseLineHighCombinedRun(int trainsize, String path) throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		String cType1 = new String("SVM");
		String cType2 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		String classifierPath4 = "/classifier_facial.arff";
		String classifierPath5 = "/classifier_edge.arff";
		

		// loading classifier instances//

		int train_size = trainsize;
		int unlabeled_size = 1000;
		int testSize = 500;
		ClassInstances classfierInstances_Sift = this.createClassInstances(path
				+ classifierPath1, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_Meta = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_Rgb = this.createClassInstances(path
				+ classifierPath3, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_facial = this.createClassInstances(
				path + classifierPath4, train_size, unlabeled_size, testSize);
		ClassInstances classfierInstances_edge = this.createClassInstances(
				path + classifierPath5, train_size, unlabeled_size, testSize);
		// //
		ClassInstances mergedIsntances = mergeInstances(
				classfierInstances_edge, classfierInstances_facial);

		ClassInstances tripleMerged = mergeInstances(mergedIsntances,
				classfierInstances_Meta);

		Instances trainSet_Sift = addInstanceSets(
				classfierInstances_Sift.getTrainInstances(),
				classfierInstances_Sift.getUnlabeledInstances());
		Instances trainSet_Meta = addInstanceSets(
				classfierInstances_Meta.getTrainInstances(),
				classfierInstances_Meta.getUnlabeledInstances());
		Instances trainSet_Rgb = addInstanceSets(
				classfierInstances_Rgb.getTrainInstances(),
				classfierInstances_Rgb.getUnlabeledInstances());
		Instances trainSet_Facial = addInstanceSets(
				classfierInstances_facial.getTrainInstances(),
				classfierInstances_facial.getUnlabeledInstances());
		Instances trainSet_Edge = addInstanceSets(
				classfierInstances_edge.getTrainInstances(),
				classfierInstances_edge.getUnlabeledInstances());

		Instances mergedHigh = addInstanceSets(
				tripleMerged.getTrainInstances(),
				tripleMerged.getUnlabeledInstances());


		double sizeF_l = 0.1;
		double sizeF_u = 0.5;

		String output = "baseLineHigh_combined" + "trainingSize_" + train_size + "_"
				+ "unlabeledSize_" + unlabeled_size;
		baseline_low_Single(mergedHigh,
				tripleMerged.getTestInstances(), cType1, sizeF_l, output,
				path);

		System.out.println("Done!\n");
	}
	
	
	private Instances addInstanceSets(Instances set1, Instances set2) {
		Instances combinedSet = set1;
		for (int i = 0; i < set2.numInstances(); i++) {
			combinedSet.add(set2.instance(i));
		}
		return combinedSet;
	}

	public void testCoTrainginRun(int size, String path) throws Exception {

		System.out.print(new Date(System.currentTimeMillis()) + "\t");
		System.out.print(new Time(System.currentTimeMillis()) + "\n");

		int no_iter = 50;
		int sample_size = 100;
		int seed = 0;

		String cType1 = new String("SVM");
		String cType2 = new String("SVM");

		String classifierPath1 = "/classifier_sift.arff";

		String classifierPath2 = "/classifier_meta.arff";

		String classifierPath3 = "/classifier_rgb.arff";

		String classifierPath4 = "/classifier_facial.arff";

		// loading classifier instances//

		int train_size = 1500;
		int unlabeled_size = size;
		int testSet = 500;
		String output = "testCoTraining:trainingSize_" + train_size + "_"
				+ "unlabeledSIze" + "_" + unlabeled_size;
		ClassInstances classfierInstances1 = this.createClassInstances(path
				+ classifierPath1, train_size, unlabeled_size, testSet);
		ClassInstances classfierInstances2 = this.createClassInstances(path
				+ classifierPath2, train_size, unlabeled_size, testSet);
		ClassInstances classfierInstances3 = this.createClassInstances(path
				+ classifierPath3, train_size, unlabeled_size, testSet);
		ClassInstances classfierInstances4 = this.createClassInstances(path
				+ classifierPath4, train_size, unlabeled_size, testSet);

		ClassInstances mergedInstances = mergeInstances(classfierInstances1,
				classfierInstances4);

		ClassInstances facial_rb_sift = mergeInstances(mergedInstances,
				classfierInstances3);
		//
		// ClassInstances mergedInstances2 = mergeInstances(classfierInstances1,
		// classfierInstances2);

		testCoTraining(facial_rb_sift.getTrainInstances(), null,
				classfierInstances2.getTrainInstances(), null,
				facial_rb_sift.getUnlabeledInstances(),
				classfierInstances2.getUnlabeledInstances(),
				facial_rb_sift.getTestInstances(),
				classfierInstances2.getTestInstances(), seed, sample_size,
				no_iter, cType1, cType2, output, path);

		System.out.println("Done!\n");
	}

	public static String getMethodName(final int depth) {
		final StackTraceElement[] ste = Thread.currentThread().getStackTrace();
		return ste[ste.length - 1 - depth].getMethodName();
	}

	public static void baseline_low_Single(Instances train_view1,
			Instances test_view1, String cType1, double sizeF_l, String output,
			String path) throws Exception {

		// FileWriter fw = new FileWriter(output);
		System.out.println(train_view1.relationName());
		String result_path = getResultLocation(train_view1.relationName());
		File dir = new File(path + result_path);
		boolean dirCreated = dir.mkdirs();
		if (!dirCreated)
			System.err.println("error creating dir " + path + result_path);
		PrintWriter fw = new PrintWriter(new FileWriter(path + result_path
				+ output), true);
		Classifier basec1 = new SMO();

		System.out.println("\nNumber of train instances: "
				+ train_view1.numInstances());
		System.out.println("Number of test instances: "
				+ test_view1.numInstances());

		// Instances train_labeled1 = Utilities.getFractionLabeled(train_view1,
		// sizeF_l, 0);
		// Instances train_labeled2 = Utilities.getFractionLabeled(train_view2,
		// sizeF_l, 0);

		if (cType1.equals("NBM")) {
			basec1 = new NaiveBayesMultinomial();
		} else if (cType1.equals("NB")) {
			basec1 = new NaiveBayes();
		} else if (cType1.equals("LR")) {
			basec1 = new Logistic();
		} else if (cType1.equals("SVM")) {
			basec1 = new SMO();
			((SMO) basec1).setBuildLogisticModels(true);
		} else if (cType1.equals("Bagging")) {
			basec1 = new Bagging();
		} else if (cType1.equals("RF")) {
			basec1 = new RandomForest();
		} else if (cType1.equals("AdaBoost")) {
			basec1 = new AdaBoostM1();
		} else {
			System.out.println("Classifier1 unknown!");
		}

		System.out.println("\nTrain!");
		/*
		 * c1.buildClassifier(train_labeled1);
		 * c2.buildClassifier(train_labeled2);
		 */

		CostSensitiveClassifier csc1 = train(train_view1, basec1);

		System.out.println("Test!");

		Evaluation eval_c1 = new Evaluation(test_view1);

		fw.write("*********************************************\n");
		fw.write("Final Predictions on the test set\n");
		eval_c1.evaluateModel(csc1, test_view1);
		fw.write("Classifier 1\n");
		fw.write(eval_c1.toClassDetailsString() + "\n");
		fw.write(eval_c1.toSummaryString() + "\n");
		fw.write(eval_c1.toMatrixString() + "\n");
		fw.write("*********************************************\n");
		System.out.println(eval_c1.toSummaryString() + "\n");
		double[][] prod = new double[test_view1.numInstances()][];

		double[] distrT_1;

		Set<String> falseNegativeurls = new HashSet<String>();
		Set<String> falsePositiveeurls = new HashSet<String>();
		Set<String> privateUrls = new HashSet<String>();
		Set<String> publicUrls = new HashSet<String>();
		
		for (int j = 0; j < test_view1.numInstances(); j++) {

			String url = map.get(chosenIndices.get(j));
			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);
			double val = testCrt.classValue();
			
			if(val == 0)
			{
				privateUrls.add(url);
				if(distrT_1[0] < distrT_1[1])
					falsePositiveeurls.add(url);
					
			}
			else
			{
				publicUrls.add(url);
				if(distrT_1[1] < distrT_1[0])
					falseNegativeurls.add(url);
			}
			prod[j] = new double[distrT_1.length];
			for (int k = 0; k < distrT_1.length; k++) {
				prod[j][k] = distrT_1[k];
			}
		}
		
		FileWriter fwUrl = new FileWriter(path +result_path+ "/falsePositiveUrls");
		for(String url : falsePositiveeurls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/falseNegativeUrls");
		for(String url : falseNegativeurls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/publicUrls");
		for(String url : publicUrls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/privateUrls");
		for(String url : privateUrls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();


		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();

		for (int j = 0; j < prod.length; j++) {

			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(
					test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double) actual_idx,
					prod[j]);
			predictions.addElement(np);
		}

		ThresholdCurve ROC = new ThresholdCurve();
		Instances curve = ROC.getCurve(predictions, 0);

		fw.write("\n\nROC for combined classifier: \n\n");
		fw.write(curve.toString());

		fw.close();
	}

	public static void baseline_low(Instances train_view1,
			Instances test_view1, Instances train_view2, Instances test_view2,
			String cType1, String cType2, double sizeF_l, String output,
			String path) throws Exception {

		// FileWriter fw = new FileWriter(output);
		String result_path = getResultLocation(train_view1.relationName()
				+ "_vs_" + train_view2.relationName());
		File dir = new File(path + result_path);
		boolean dirCreated = dir.mkdirs();
		if (!dirCreated)
			System.err.println("error creating dir " + path + result_path);

		PrintWriter fw = new PrintWriter(new FileWriter(path + result_path
				+ output), true);
		PrintWriter fwFinal = new PrintWriter(new FileWriter(path + result_path
				+ "finalStats", true), true);
		Classifier basec1 = new SMO();
		Classifier basec2 = new SMO();

		System.out.println("\nNumber of train instances: "
				+ train_view1.numInstances());
		System.out.println("Number of test instances: "
				+ test_view1.numInstances());

		// Instances train_labeled1 = Utilities.getFractionLabeled(train_view1,
		// sizeF_l, 0);
		// Instances train_labeled2 = Utilities.getFractionLabeled(train_view2,
		// sizeF_l, 0);

		if (cType1.equals("NBM")) {
			basec1 = new NaiveBayesMultinomial();
		} else if (cType1.equals("NB")) {
			basec1 = new NaiveBayes();
		} else if (cType1.equals("LR")) {
			basec1 = new Logistic();
		} else if (cType1.equals("SVM")) {
			basec1 = new SMO();
			((SMO) basec1).setBuildLogisticModels(true);
		} else if (cType1.equals("Bagging")) {
			basec1 = new Bagging();
		} else if (cType1.equals("RF")) {
			basec1 = new RandomForest();
		} else if (cType1.equals("AdaBoost")) {
			basec1 = new AdaBoostM1();
		} else {
			System.out.println("Classifier1 unknown!");
		}

		if (cType2.equals("NBM")) {
			basec2 = new NaiveBayesMultinomial();
		} else if (cType2.equals("NB")) {
			basec2 = new NaiveBayes();
		} else if (cType2.equals("LR")) {
			basec2 = new Logistic();
		} else if (cType2.equals("SVM")) {
			basec2 = new SMO();
			((SMO) basec2).setBuildLogisticModels(true);
		} else if (cType2.equals("Bagging")) {
			basec2 = new Bagging();
		} else if (cType2.equals("RF")) {
			basec2 = new RandomForest();
		} else if (cType2.equals("AdaBoost")) {
			basec2 = new AdaBoostM1();
		} else {
			System.out.println("Classifier2 unknown!");
		}

		System.out.println("\nTrain!");
		/*
		 * c1.buildClassifier(train_labeled1);
		 * c2.buildClassifier(train_labeled2);
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

		for (int j = 0; j < test_view1.numInstances(); j++) {

			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);

			prod[j] = new double[distrT_1.length];
			for (int k = 0; k < distrT_1.length; k++) {
				prod[j][k] = distrT_1[k];
			}
		}

		double[] distrT_2;
		Set<String> falseNegativeurls = new HashSet<String>();
		Set<String> falsePositiveeurls = new HashSet<String>();
		Set<String> privateUrls = new HashSet<String>();
		Set<String> publicUrls = new HashSet<String>();
		for (int j = 0; j < test_view2.numInstances(); j++) {

			String url = map.get(chosenIndices.get(j));
			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);

			for (int k = 0; k < distrT_2.length; k++) {
				prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				// prod[j][k] += distrT_2[k];
			}
			
			double val = testCrt.classValue();
			if(val == 0)
			{
				privateUrls.add(url);
				if(prod[j][0] < prod[j][1])
					falsePositiveeurls.add(url);
					
			}
			else
			{
				publicUrls.add(url);
				if(prod[j][1] < prod[j][0])
					falseNegativeurls.add(url);
			}
			
		}

		
		
		FileWriter fwUrl = new FileWriter(path +result_path+ "/falsePositiveUrls");
		for(String url : falsePositiveeurls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/falseNegativeUrls");
		for(String url : falseNegativeurls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/publicUrls");
		for(String url : publicUrls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		fwUrl = new FileWriter(path +result_path+ "/privateUrls");
		for(String url : privateUrls)
		{
			fwUrl.write(url + "\n");
		}
		fwUrl.close();
		
		
		for (int j = 0; j < prod.length; j++) {

			if (!Utils.eq(Utils.sum(prod[j]), 0)) {
				Utils.normalize(prod[j]);
			}
		}

		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();

		for (int j = 0; j < prod.length; j++) {

			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(
					test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double) actual_idx,
					prod[j]);
			predictions.addElement(np);
		}

		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");

		System.out.println("*********************************************\n");
		System.out.println("Size - : " + train_view1.numInstances());
		System.out.println(eval.toSummaryString() + "\n");
		System.out.println("*********************************************\n");

		fwFinal.write("*********************************************\n");
		fwFinal.write("\nPredictions for the Combined Classifier!");
		fwFinal.write("Size - : " + train_view1.numInstances());
		fwFinal.write(eval.toSummaryString() + "\n");
		fwFinal.write("*********************************************\n");

		ThresholdCurve ROC = new ThresholdCurve();
		Instances curve = ROC.getCurve(predictions, 0);

		fw.write("\n\nROC for combined classifier: \n\n");
		fw.write(curve.toString());

		fw.close();
		fwFinal.close();
	}

	public static void testCoTraining(Instances train_view1,
			Instances valid_view1, Instances train_view2,
			Instances valid_view2, Instances unlabeled_view1,
			Instances unlabeled_view2, Instances test_view1,
			Instances test_view2, int seed, int sample_size, int no_iter,
			String cType1, String cType2, String output, String path)
			throws Exception {

		// FileWriter fw = new FileWriter(output);
		// output = getMethodName(2) + "_" + output;
		int unlabSize = unlabeled_view1.numInstances();
		String result_path = getResultLocation(train_view1.relationName()
				+ "_vs_" + train_view2.relationName());
		File dir = new File(path + result_path);
		boolean dirCreated = dir.mkdirs();
		if (!dirCreated)
			System.err.println("error creating dir " + path + result_path);
		PrintWriter fw = new PrintWriter(new FileWriter(path + result_path
				+ output), true);
		PrintWriter fwFinal = new PrintWriter(new FileWriter(path + result_path
				+ "finalStats", true), true);
		CostSensitiveClassifier csc1 = null;
		CostSensitiveClassifier csc2 = null;

		Random random;

		System.out.println("\nNumber of labeled instances: "
				+ train_view1.numInstances());
		fw.write("Number of labeled instances: " + train_view1.numInstances()
				+ "\n\n");

		System.out.println("\nNumber of unlabeled instances: "
				+ unlabeled_view1.numInstances());
		fw.write("Number of unlabeled instances: "
				+ unlabeled_view1.numInstances() + "\n\n");

		int[] ratios = Utilities.getExpRatios(train_view1);
		// int[] ratios = {2,2};

		Instances labeled_view1 = new Instances(train_view1);
		Instances labeled_view2 = new Instances(train_view2);

		// create the header for u_prime1 and u_prime2
		Instances u_prime1 = new Instances(unlabeled_view1);
		u_prime1.delete();

		Instances u_prime2 = new Instances(unlabeled_view2);
		u_prime2.delete();

		// sampling u_prime1 and u_prime2
		random = new Random(seed);
		for (int n_u = 0; n_u < sample_size
				&& unlabeled_view1.numInstances() > 0; n_u++) {
			int index = random.nextInt(unlabeled_view2.numInstances());

			Instance inst1 = unlabeled_view1.instance(index);
			u_prime1.add(inst1);
			unlabeled_view1.delete(index);

			Instance inst2 = unlabeled_view2.instance(index);
			u_prime2.add(inst2);
			unlabeled_view2.delete(index);
		} // done sampling u_prime1 and u_prime2

		// sum is the number of elements to be added to labeled data at each
		// iteration
		int sum = Utils.sum(ratios);
		int s = 2 * sum; // the number of examples extracted at each iteration
							// from each view
		// no_iter = (int) Math.floor((double)unlabeled_view_1.numInstances() /
		// (double)s);
		// System.out.println("no_iter: "+no_iter);

		int crt_iter = 0; // counts "good" iterations -
							// when the right number of
							// instances can be added to labeled

		boolean enoughInstances = true; // makes sure that
										// there are at least U'
										// sample_size instances
										// left in unlabeled

		double[] class_distr_1;
		double[] class_distr_2;

		// ea stands for entry array
		// stores instances in U' and their predictions
		// used to make sorting easy

		Entry[] ea_1 = null;
		Entry[] ea_2 = null;

		// arrays of indices of instances
		// that need to be added to the training
		ArrayList<Integer> al_1 = null;
		ArrayList<Integer> al_2 = null;

		// index_ea_i (i=1,2) has one entry for each class
		// tells us what's the last instance that we used
		// in the list that will be added to labeled
		int[] index_ea_1 = null;
		int[] index_ea_2 = null;

		int[] indices_1 = null;
		int[] indices_2 = null;
		int[] indices = null;

		int len_1 = 0;
		int len_2 = 0;
		int len = 0;

		Classifier basec1 = new SMO();
		Classifier basec2 = new SMO();

		while ((crt_iter < no_iter) && enoughInstances) {

			System.out.println("Number of labeled instances: "
					+ labeled_view1.numInstances());
			fw.write("Number of labeled instances: "
					+ labeled_view1.numInstances() + "\n\n");

			if (cType1.equals("NBM")) {
				basec1 = new NaiveBayesMultinomial();
			} else if (cType1.equals("NB")) {
				basec1 = new NaiveBayes();
			} else if (cType1.equals("LR")) {
				basec1 = new Logistic();
			} else if (cType1.equals("SVM")) {
				basec1 = new SMO();
				((SMO) basec1).setBuildLogisticModels(true);
			} else if (cType1.equals("Bagging")) {
				basec1 = new Bagging();
			} else if (cType1.equals("RF")) {
				basec1 = new RandomForest();
			} else if (cType1.equals("AdaBoost")) {
				basec1 = new AdaBoostM1();
			} else {
				System.out.println("Classifier1 unknown!");
			}

			if (cType2.equals("NBM")) {
				basec2 = new NaiveBayesMultinomial();
			} else if (cType2.equals("NB")) {
				basec2 = new NaiveBayes();
			} else if (cType2.equals("LR")) {
				basec2 = new Logistic();
			} else if (cType2.equals("SVM")) {
				basec2 = new SMO();
				((SMO) basec2).setBuildLogisticModels(true);
			} else if (cType2.equals("Bagging")) {
				basec2 = new Bagging();
			} else if (cType2.equals("RF")) {
				basec2 = new RandomForest();
			} else if (cType2.equals("AdaBoost")) {
				basec2 = new AdaBoostM1();
			} else {
				System.out.println("Classifier2 unknown!");
			}

			System.out.println("\nTrain on iteration: " + crt_iter);
			/*
			 * //learn classifiers from the set of labeled instances //c1 = new
			 * SMO(); //((SMO)c1).setBuildLogisticModels(true);
			 * c1.buildClassifier(labeled_view1);
			 * 
			 * //c2 = new SMO(); //((SMO)c2).setBuildLogisticModels(true);
			 * c2.buildClassifier(labeled_view2);
			 */

			csc1 = train(labeled_view1, basec1);
			csc2 = train(labeled_view2, basec2);

			if (crt_iter == 0) {
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

			if (valid_view1 != null && valid_view2 != null) {

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

			// foundFlag is true when instances in the right classes are found
			// among the predicted instances (in both views)

			boolean found1 = false;
			boolean found2 = false;

			boolean diff = true;

			while (diff && enoughInstances) {
				while ((!found1 || !found2) && enoughInstances) {
					found1 = true;
					found2 = true;

					ea_1 = new Entry[u_prime1.numInstances()];
					ea_2 = new Entry[u_prime2.numInstances()];

					al_1 = new ArrayList<Integer>();
					al_2 = new ArrayList<Integer>();

					index_ea_1 = new int[labeled_view1.numClasses()];
					index_ea_2 = new int[labeled_view2.numClasses()];

					// classify u_prime1 with the classifier c1
					for (int j = 0; j < u_prime1.numInstances(); j++) {

						Instance u_prime1_j = u_prime1.instance(j);

						class_distr_1 = csc1
								.distributionForInstance(u_prime1_j);
						double predicted = Utils.maxIndex(class_distr_1);
						String value = u_prime1.classAttribute().value(
								(int) predicted);

						// add class probabilities together with instance
						// indices to ea_1
						ea_1[j] = new Entry(j, class_distr_1[(int) predicted]);

						u_prime1_j.setClassMissing();
						u_prime1_j.setClassValue(value);
					}

					// ea_1 contains class probability and instance index
					// sort ea_1 after class probabilities (increasing order)
					Arrays.sort(ea_1, 0, u_prime1.numInstances(), ea_1[0]);

					// find examples to add to the training set
					for (int j = 0; j < ratios.length; j++) {
						int n = ratios[j];
						int found = 0;

						// search in ea_1 from the end to the beginning
						for (int idx = ea_1.length - 1; idx >= 0; idx--) {

							// get the index of the current instance (second
							// comp in ea_1 - denoted by .i)
							int crt_idx = ea_1[idx].i;

							// get the instance with this best prediction and
							// check its class
							Instance crt = u_prime1.instance(crt_idx);
							String cls = crt.stringValue(crt.classAttribute());

							// if cls is the class we are looking for
							if (cls.equals(labeled_view1.classAttribute()
									.value(j))) {
								// add the index of that instance to the array
								// list
								al_1.add(crt_idx);
								found++;

								// if we don't need instances in this class
								// anymore,
								// remember where we left for that class
								if (found >= n) {
									index_ea_1[j] = idx;
									break;
								}
							}
						}
						if (found < ratios[j]) {
							found1 = false;
							break;
						}
					}

					// classify u_prime2 with the classifier c2
					for (int j = 0; j < u_prime2.numInstances(); j++) {

						Instance u_prime2_j = u_prime2.instance(j);

						class_distr_2 = csc2
								.distributionForInstance(u_prime2_j);
						double predicted = Utils.maxIndex(class_distr_2);
						String value = u_prime2.classAttribute().value(
								(int) predicted);

						// add class probabilities together with instance
						// indices to ea_2
						ea_2[j] = new Entry(j, class_distr_2[(int) predicted]);

						u_prime2_j.setClassMissing();
						u_prime2_j.setClassValue(value);
					}

					// sort ea_2 after class probabilities (increasing order)
					Arrays.sort(ea_2, 0, u_prime2.numInstances(), ea_2[0]);

					for (int j = 0; j < ratios.length; j++) {
						int n = ratios[j];
						int found = 0;

						for (int idx = ea_2.length - 1; idx >= 0; idx--) {
							int crt_idx = ea_2[idx].i;

							Instance crt = u_prime2.instance(crt_idx);
							String cls = crt.stringValue(crt.classAttribute());
							if (cls.equals(labeled_view2.classAttribute()
									.value(j))) {

								al_2.add(crt_idx);
								found++;
								if (found >= n) {
									index_ea_2[j] = idx;
									break;
								}
							}
						}
						if (found < ratios[j]) {
							found2 = false;
							break;
						}
					}

					if (!found1 || !found2) {
						if (sample_size > unlabeled_view1.numInstances()) {
							enoughInstances = false;
						} else {
							// sample from unlabeled until double U'
							random = new Random(seed);
							for (int n_u = 0; n_u < sample_size
									&& unlabeled_view1.numInstances() > 0; n_u++) {
								int index = random.nextInt(unlabeled_view1
										.numInstances());

								Instance inst1 = unlabeled_view1
										.instance(index);
								u_prime1.add(inst1);
								unlabeled_view1.delete(index);

								Instance inst2 = unlabeled_view2
										.instance(index);
								u_prime2.add(inst2);
								unlabeled_view2.delete(index);
							}
						}
					}
				} // end while((!found1 || !found2) && enoughInstances)

				// sort indices to find out if there are instances that
				// have conflicting labels in the two views
				len_1 = al_1.size();
				indices_1 = new int[len_1];
				for (int j = 0; j < len_1; j++)
					indices_1[j] = al_1.get(j);
				Arrays.sort(indices_1, 0, len_1);

				len_2 = al_2.size();
				indices_2 = new int[len_2];
				for (int j = 0; j < len_2; j++)
					indices_2[j] = al_2.get(j);
				Arrays.sort(indices_2, 0, len_2);

				len = len_1 + len_2;
				indices = new int[len];
				for (int j = 0; j < len_1; j++)
					indices[j] = indices_1[j];
				for (int j = 0; j < len_2; j++)
					indices[j + len_1] = indices_2[j];
				Arrays.sort(indices, 0, len);

				diff = false;
				for (int j = len - 1; j > 0; j--) {

					// check if an instance is in both lists
					if (indices[j] == indices[j - 1]) {

						Instance crt_1 = u_prime1.instance(indices[j]);
						String cls_1 = crt_1
								.stringValue(crt_1.classAttribute());
						int cls_idx_1 = (int) crt_1.value(crt_1
								.classAttribute());

						Instance crt_2 = u_prime2.instance(indices[j]);
						String cls_2 = crt_2
								.stringValue(crt_2.classAttribute());
						int cls_idx_2 = (int) crt_2.value(crt_2
								.classAttribute());

						if (!cls_1.equals(cls_2)) {

							diff = true;

							// remove instances whose classes are different
							// from the two array lists al_1 and al_2
							al_1.remove((Integer) indices[j]);
							al_2.remove((Integer) indices[j]);

							found1 = false;
							found2 = false;

							// continue to search for another non-conflicting
							// example from where we left and keep track of
							// the new index in the corresponding class

							for (int idx = index_ea_1[cls_idx_1] - 1; idx >= 0; idx--) {
								int crt_idx = ea_1[idx].i;

								Instance crt = u_prime1.instance(crt_idx);
								String cls = crt.stringValue(crt
										.classAttribute());
								if (cls.equals(cls_1)) {
									al_1.add(crt_idx);
									index_ea_1[cls_idx_1] = idx;
									found1 = true;
									break;
								}
							}

							for (int idx = index_ea_2[cls_idx_2] - 1; idx >= 0; idx--) {
								int crt_idx = ea_2[idx].i;

								Instance crt = u_prime2.instance(crt_idx);
								String cls = crt.stringValue(crt
										.classAttribute());
								if (cls.equals(cls_2)) {
									al_2.add(crt_idx);
									index_ea_2[cls_idx_2] = idx;
									found2 = true;
									break;
								}
							}

							if (!found1 || !found2)
								break;
						} // end if (!cls_1.equals(cls_2))
						j = j - 1;
					}
				}

				if (!found1 || !found2) {
					if (sample_size > unlabeled_view1.numInstances()) {
						enoughInstances = false;
					} else {
						random = new Random(seed);
						// sample from unlabeled until double U'
						for (int n_u = 0; n_u < sample_size
								&& unlabeled_view1.numInstances() > 0; n_u++) {
							int index = random.nextInt(unlabeled_view1
									.numInstances());

							Instance inst1 = unlabeled_view1.instance(index);
							u_prime1.add(inst1);
							unlabeled_view1.delete(index);

							Instance inst2 = unlabeled_view2.instance(index);
							u_prime2.add(inst2);
							unlabeled_view2.delete(index);
						}
					}
				}
			} // end while (diff) loop

			if (enoughInstances) {
				// sort indices - we will delete them from the last to the first
				len_1 = al_1.size();
				indices_1 = new int[len_1];
				for (int j = 0; j < len_1; j++)
					indices_1[j] = al_1.get(j);
				Arrays.sort(indices_1, 0, len_1);

				len_2 = al_2.size();
				indices_2 = new int[len_2];
				for (int j = 0; j < len_2; j++)
					indices_2[j] = al_2.get(j);
				Arrays.sort(indices_2, 0, len_2);

				len = len_1 + len_2;
				indices = new int[len];
				for (int j = 0; j < len_1; j++)
					indices[j] = indices_1[j];
				for (int j = 0; j < len_2; j++)
					indices[j + len_1] = indices_2[j];
				Arrays.sort(indices, 0, len);

				System.out.println("Done with iter " + crt_iter
						+ " - adding examples for next iter");
				crt_iter++;
				// add best predicted in view 1
				System.out.println("Instances added to view2");
				for (int j = len_1 - 1; j >= 0; j--) {
					// find best predicted instances and their class in view 1
					Instance crt_1 = u_prime1.instance(indices_1[j]);
					String cls_1 = crt_1.stringValue(crt_1.classAttribute());
					// switch to view 2
					Instance crt_2 = u_prime2.instance(indices_1[j]);

					crt_2.setClassMissing();
					crt_2.setClassValue(cls_1);

					labeled_view2.add(crt_2);
					// labeled_view1.add(crt_1);
				}

				System.out.println("Instances added to view1");
				// add best predicted in view 2
				for (int j = len_2 - 1; j >= 0; j--) {
					// find best predicted instances and their class in view 2
					Instance crt_2 = u_prime2.instance(indices_2[j]);
					String cls_2 = crt_2.stringValue(crt_2.classAttribute());
					// switch to view 1
					Instance crt_1 = u_prime1.instance(indices_2[j]);

					crt_1.setClassMissing();
					crt_1.setClassValue(cls_2);

					labeled_view1.add(crt_1);
					// labeled_view2.add(crt_2);
				}

				// delete examples added to training from both views
				for (int j = len - 1; j > 0; j--) {
					if (indices[j] == indices[j - 1]) {
						u_prime1.delete(indices[j]);
						u_prime2.delete(indices[j]);
						j = j - 1;
					} else {
						u_prime1.delete(indices[j]);
						u_prime2.delete(indices[j]);

						if ((j - 1) == 0) {
							u_prime1.delete(indices[j - 1]);
							u_prime2.delete(indices[j - 1]);
						}
					}
				}

				// add 2*ratio examples to U' in both views
				if (s > unlabeled_view1.numInstances()) {
					enoughInstances = false;
				} else {
					// sample from unlabeled to add to U' s examples
					random = new Random(seed);
					for (int n_u = 0; n_u < s; n_u++) {
						int index = random.nextInt(unlabeled_view1
								.numInstances());

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

		csc1 = train(labeled_view1, basec1);
		csc2 = train(labeled_view2, basec2);

		double[][] prod = new double[test_view1.numInstances()][];

		double[] distrT_1;

		for (int j = 0; j < test_view1.numInstances(); j++) {

			Instance testCrt = test_view1.instance(j);
			distrT_1 = csc1.distributionForInstance(testCrt);

			prod[j] = new double[distrT_1.length];
			for (int k = 0; k < distrT_1.length; k++) {
				prod[j][k] = distrT_1[k];
			}
		}

		double[] distrT_2;

		for (int j = 0; j < test_view2.numInstances(); j++) {

			Instance testCrt = test_view2.instance(j);
			distrT_2 = csc2.distributionForInstance(testCrt);

			for (int k = 0; k < distrT_2.length; k++) {
				// prod[j][k] *= distrT_2[k];
				// for taking the average instead of product
				prod[j][k] += distrT_2[k];
			}
		}

		for (int j = 0; j < prod.length; j++) {

			if (!Utils.eq(Utils.sum(prod[j]), 0)) {
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

		fwFinal.write("*********************************************\n");
		fwFinal.write("Number of labeled instances: "
				+ train_view1.numInstances() + "\n\n");

		fwFinal.write("Number of unlabeled instances: " + unlabSize + "\n\n");

		fwFinal.write("Number of test instances: " + test_view1.numInstances()
				+ "\n\n");

		fwFinal.write("Final Predictions on the test set\n");
		eval_c1.evaluateModel(csc1, test_view1);
		fwFinal.write("Classifier 1\n");
		fwFinal.write(eval_c1.toClassDetailsString() + "\n");
		fwFinal.write(eval_c1.toSummaryString() + "\n");
		fwFinal.write(eval_c1.toMatrixString() + "\n");
		eval_c2.evaluateModel(csc2, test_view2);
		fwFinal.write("Classifier 2\n");
		fwFinal.write(eval_c2.toClassDetailsString() + "\n");
		fwFinal.write(eval_c2.toSummaryString() + "\n");
		fwFinal.write(eval_c2.toMatrixString() + "\n");
		fwFinal.write("\n");

		Evaluation_D eval = new Evaluation_D(test_view1);
		FastVector predictions = new FastVector();

		for (int j = 0; j < prod.length; j++) {

			eval.updateStatsForClassifier(prod[j], test_view1.instance(j));

			String actual = test_view1.instance(j).stringValue(
					test_view1.instance(j).classAttribute());
			int actual_idx = test_view1.classAttribute().indexOfValue(actual);
			NominalPrediction np = new NominalPrediction((double) actual_idx,
					prod[j]);
			predictions.addElement(np);
		}

		fw.write("*********************************************\n");
		fw.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fw.write(eval.toClassDetailsString() + "\n");
		fw.write(eval.toSummaryString() + "\n");
		fw.write(eval.toMatrixString() + "\n");
		fw.write("*********************************************\n");

		fwFinal.write("\nPredictions for the Combined Classifier!");
		eval.setPredictions(predictions);
		fwFinal.write(eval.toClassDetailsString() + "\n");
		fwFinal.write(eval.toSummaryString() + "\n");
		fwFinal.write(eval.toMatrixString() + "\n");
		fwFinal.write("*********************************************\n");

		ThresholdCurve ROC = new ThresholdCurve();
		Instances curve = ROC.getCurve(predictions, 0);

		fw.write("\n\nROC for combined classifier: \n\n");
		fw.write(curve.toString());

		fw.close();
		fwFinal.close();

	}

	public static CostSensitiveClassifier train(Instances train,
			Classifier classifier) throws Exception {
		train.setClassIndex(train.numAttributes() - 1);

		double[] counts = new double[2];
		counts[0] = 0;
		counts[1] = 0;
		for (int tx = 0; tx < train.numInstances(); tx++) {
			// System.out.println(train.instance(tx).classValue());
			// System.out.println(train.instance(tx).stringValue(train.numAttributes()
			// - 1));
			// System.out.println(train.instance(tx));
			if (train.instance(tx).classValue() == 0)
				counts[0]++;
			else
				counts[1]++;
		}

		CostMatrix cmatrix = new CostMatrix(2);
		cmatrix.setCell(0, 0, (double) 0);
		cmatrix.setCell(1, 1, (double) 0);
		cmatrix.setCell(0, 1, (double) 1);
		cmatrix.setCell(1, 0, (double) 1);

		if (counts[1] > counts[0]) // neg instances are more then cost of fn is
									// more
		{
			cmatrix.setCell(0, 1, (double) (counts[1] / counts[0]));
		} else if (counts[0] > counts[1]) {
			cmatrix.setCell(1, 0, (double) (counts[0] / counts[1]));
		}

		System.out.println("pos counts:" + counts[0] + " neg counts:"
				+ counts[1]);
		System.out.println("Cost matrix " + cmatrix);

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

}
