package cotraining;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

/**
 * performs attribute selection using CfsSubsetEval and GreedyStepwise
 * (backwards) and trains J48 with that. Needs 3.5.5 or higher to compile.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AttributeSelectionTest {

  /**
   * uses the meta-classifier
   */
  protected static void useClassifier(Instances data) throws Exception {
    System.out.println("\n1. Meta-classfier");
    AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
    CfsSubsetEval eval = new CfsSubsetEval();
    GreedyStepwise search = new GreedyStepwise();
    search.setSearchBackwards(true);
    J48 base = new J48();
    classifier.setClassifier(base);
    classifier.setEvaluator(eval);
    classifier.setSearch(search);
    Evaluation evaluation = new Evaluation(data);
    evaluation.crossValidateModel(classifier, data, 10, new Random(1));
    System.out.println(evaluation.toSummaryString());
  }

  /**
   * uses the filter
   */
  protected static void useFilter(Instances data) throws Exception {
    System.out.println("\n2. Filter");
    weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
    CfsSubsetEval eval = new CfsSubsetEval();
    GreedyStepwise search = new GreedyStepwise();
    search.setSearchBackwards(true);
    filter.setEvaluator(eval);
    filter.setSearch(search);
    filter.setInputFormat(data);
    Instances newData = Filter.useFilter(data, filter);
    System.out.println(newData);
  }

  /**
   * uses the low level approach
   */
  protected static int[] useLowLevel(Instances data) throws Exception {
    System.out.println("\n3. Low-level");
    AttributeSelection attsel = new AttributeSelection();
    CfsSubsetEval eval = new CfsSubsetEval();
    GreedyStepwise search = new GreedyStepwise();
    search.setSearchBackwards(true);
    attsel.setEvaluator(eval);
    attsel.setSearch(search);
    attsel.SelectAttributes(data);
    int[] indices = attsel.selectedAttributes();
    System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices));
    return indices;
  }

	public static List<String> getTagSet(File mdFile) {
		FileInputStream inputStream;
		try {
			inputStream = new FileInputStream(mdFile);
			DataInputStream in = new DataInputStream(inputStream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			StringBuffer url = new StringBuffer(2000);
			List<String> set = new ArrayList<String>();
			while ((url.append(br.readLine())).toString() != null) {
				String[] starr = url.toString().split("-");
				String keyval = "";
				for (int i = 1; i < starr.length; i++)
					keyval += starr[i];

				String[] kvp = keyval.split("jpg:");
				if (kvp.length == 1)
					break;
				String[] tgs_privcy = kvp[1].split("_");
				
				String[] tags = tgs_privcy[0].split(",");
				for(String tag: tags)
				{
					tag = tag.trim();
					set.add(tag);
				}
				url.delete(0, url.length() - 1);
			}
			return set;
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;

	}
  
  /**
   * takes a dataset as first argument
   *
   * @param args        the commandline arguments
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // load data
    System.out.println("\n0. Loading data");
    String path = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/september9/10000/arff_facial/";
    String classifierPath = path+"/classifier_meta_1.arff";
    String mdFile = path+"10000_unique_metadata";
    List<String> tags = getTagSet(new File(mdFile));
    FileWriter tagWriter = new FileWriter(path + "/AllWords");
    for(String str : tags)
    {
    	tagWriter.write(str + "\n");
    }
    tagWriter.close();

    DataSource source = new DataSource(classifierPath);
    Instances data = source.getDataSet();
    if (data.classIndex() == -1)
      data.setClassIndex(data.numAttributes() - 1);

    HashMap<String, Integer> map = new HashMap<String, Integer>();
    HashMap<Integer, Set<String> > revMap = new HashMap<Integer, Set<String>>();
    
    for(String str: tags)
    {
    	if(! map.containsKey(str))
    	{
    		map.put(str, 1);
    		Set<String> set = null;
    		if(revMap.containsKey(1))
    		{
    			set = revMap.get(1);
    		}
    		else
    		{
    			set = new HashSet<String>();
    		}
    		set.add(str);
    		revMap.put(1, set);
    	
    	}
    	else
    	{
    		int c = map.get(str);
    		Set<String> oldSet = revMap.get(c);
    		if(oldSet != null)
    		{
    			oldSet.remove(str);
    			if(oldSet.size() > 0)
    				revMap.put(c, oldSet);
    			else
    				revMap.remove(c);
    					
    		}
    		map.put(str, c+1);
    		Set<String> set = null;
    		if(revMap.containsKey(c+1))
    		{
    			set = revMap.get(c+1);
    		}
    		else
    		{
    			set = new HashSet<String>();
    		}
    		set.add(str);
    		revMap.put(c+1, set);
    	}
    }
    FileWriter fwUrl = new FileWriter(path + "/wordFrequencies");
    Integer[] intArr = revMap.keySet().toArray(new Integer[0]);
    for(int i = intArr.length -1; i >= 0 ; i-- )
    {
    	if(revMap.containsKey(i))
    	{
	    	Set<String> strSet = revMap.get(intArr[i]);
	    	fwUrl.write("frequency - " + i + ": ");
	    	System.out.println("frequency - " + i);
	    	for(String str : strSet)
	    	{
	    		System.out.print(str + ", ");
	    		fwUrl.write(str + ", ");
	    	}
	    	System.out.println("");
	    	fwUrl.write("\n");
    	}

    }
	fwUrl.close();
//    // 1. meta-classifier
//    useClassifier(data);
//
//    // 2. filter
//    useFilter(data);

//    // 3. low-level
//    int [] indices = useLowLevel(data);
//    String[] tagArr = tags.toArray(new String[0]);
//    for(int i : indices)
//    	System.out.println(tagArr[i]);
//    
  }
  
  
  
  
}
