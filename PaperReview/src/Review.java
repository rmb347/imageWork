import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Set;


public class Review {
	
	private static String cosineOpFile;
	private static String pearsonOpFile;

	public static void main(String argsp[])
	{
		cosineReview();

		//pearsonReview();

	}

	private static void pearsonReview() {
		Scanner sc = new Scanner(System.in);
		HashMap<String, Integer[] > userResponseMap = new HashMap<String, Integer[] >();
		System.out.print ("Input file : ");
		String fileName = sc.next();
		System.out.print ("Pearson Output file : ");
		pearsonOpFile = sc.next();
		File file = new File(fileName);
		if (file.exists()) {
			try {
				FileInputStream inputStream = new FileInputStream(file);
				DataInputStream in = new DataInputStream(inputStream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String line = null;
				
				while((line = br.readLine()) != null)
				{
					String[] responses = line.split(",");
					String userID = responses[responses.length-1];
					Integer[] resposneArray = getIntResponseArray(responses);
					userResponseMap.put(userID, resposneArray);
				}
				
				processMapPearson(userResponseMap);
				
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
	}

	private static void processMapPearson(HashMap<String, Integer[] > userResponseMap)
	{
		createHeader(userResponseMap.keySet(), new File(pearsonOpFile));
		for(String user : userResponseMap.keySet())
		{
			getBestCompatibleUser_pearson(userResponseMap, user);
		}
	}
	
	
	private static void getBestCompatibleUser_pearson(
			HashMap<String, Integer[]> userResponseMap, String user) {
		FileWriter writer = null;
		double max = Double.MIN_VALUE;
		try {
			writer = new FileWriter(pearsonOpFile,true);
			String maxuser = null;
			writer.write(user + ", ");
			for (String usr : userResponseMap.keySet()) {
				if (!usr.equals(user)) {
					try {
						double res = caluculateCompatibility_pearson(userResponseMap.get(usr),
								userResponseMap.get(user));
						System.out.println(res);
						if(max < res)
						{
							max =res;
							maxuser = usr;
						}
						writer.write(res + ", ");
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				else
				{
					writer.write(" - ,");
				}

			}
			System.out.println("Best user for " + user + " is :" + maxuser);
			writer.write("\n");
			writer.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		//Input file : /Users/Aurnob/Desktop/cosine_rvw.csv
		//Output file : /Users/Aurnob/Desktop/cosine_op.csv
		//Input file : /Users/Aurnob/Desktop/pearson_rvw.cs			/Users/Aurnob/Desktop/pearson_op.csv
		//pearson_rvw.csv
	}

	private static double caluculateCompatibility_pearson(Integer[] responses1,
			Integer[] responses2) throws Exception {
		
		double ave1=0;
		for(int i = 0; i < responses1.length; i++)
		{
			ave1 += responses1[i] ;
		}
		ave1 = ave1/ responses1.length;
		
		double ave2=0;
		for(int i = 0; i < responses2.length; i++)
		{
			ave2 += responses2[i] ;
		}
		ave2 = ave2/ responses2.length;
		
		double num  = 0;
		
		for(int i = 0; i < responses1.length; i++)
		{
			num += (responses1[i]-ave1) *(responses2[i] - ave2);
		}
		
		double den1 = 0;
		for(int i = 0; i < responses1.length; i++)
		{
			den1 += (responses1[i]-ave1)*(responses1[i]-ave1);
		}
		
		double den2 = 0;
		for(int i = 0; i < responses1.length; i++)
		{
			den2 += (responses2[i] - ave2)*(responses2[i] - ave2);
		}
		
		double den = Math.sqrt(den1)*Math.sqrt(den2);
		
		if(den != 0)
		{
			return (num/den);
		}
		else
		{
			throw new Exception("denominator is 0");
		}
	}

	private static void cosineReview() {
		Scanner sc = new Scanner(System.in);
		HashMap<String, Integer[] > userResponseMap = new HashMap<String, Integer[] >();
		System.out.print ("Input file : ");
		String fileName = sc.next();
		System.out.print ("Cosine Output file : ");
		cosineOpFile = sc.next();
		File file = new File(fileName);
		if (file.exists()) {
			try {
				FileInputStream inputStream = new FileInputStream(file);
				DataInputStream in = new DataInputStream(inputStream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String line = null;
				
				while((line = br.readLine()) != null)
				{
					String[] responses = line.split(",");
					String userID = responses[responses.length-1];
					Integer[] resposneArray = getIntResponseArray(responses);
					userResponseMap.put(userID, resposneArray);
				}
				
				processMapCosine(userResponseMap);
				
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
	}

	private static void processMapCosine(HashMap<String, Integer[] > userResponseMap)
	{
		createHeader(userResponseMap.keySet(), new File(cosineOpFile));
		for(String user : userResponseMap.keySet())
		{
			getBestCompatibleUser_cosine(userResponseMap, user);
		}
	}
	
	private static void createHeader(Set<String> keySet, File file) {
		FileWriter writer = null;
		try {
			writer = new FileWriter(file,true);
			writer.write("USER _ ID ,");
			for(String s : keySet)
			{
				writer.write(s + ",");
			}
			writer.write("\n");
			writer.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}

	}

	private static void getBestCompatibleUser_cosine(
			HashMap<String, Integer[]> userResponseMap, String user) {
		FileWriter writer = null;
		double max = Double.MIN_VALUE;
		try {
			writer = new FileWriter(cosineOpFile,true);
			String maxuser = null;
			writer.write(user + ", ");
			for (String usr : userResponseMap.keySet()) {
				if (!usr.equals(user)) {
					try {
						double res = caluculateCompatibility_cosine(userResponseMap.get(usr),
								userResponseMap.get(user));
						System.out.println(res);
						if(max < res)
						{
							max =res;
							maxuser = usr;
						}
						writer.write(res + ", ");
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				else
				{
					writer.write(" - ,");
				}

			}
			System.out.println("Best user for " + user + " is :" + maxuser);
			writer.write("\n");
			writer.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
//Input file : /Users/Aurnob/Desktop/cosine_rvw.csv
		//Output file : /Users/Aurnob/Desktop/cosine_op.csv


	}

	private static double caluculateCompatibility_cosine(Integer[] responses1,
			Integer[] responses2) throws Exception {
		double num  = 0;
		
		for(int i = 0; i < responses1.length; i++)
		{
			num += responses1[i] *responses2[i];
		}
		
		double den1 = 0;
		for(int i = 0; i < responses1.length; i++)
		{
			den1 += responses1[i]*responses1[i];
		}
		
		double den2 = 0;
		for(int i = 0; i < responses1.length; i++)
		{
			den2 += responses2[i]*responses2[i];
		}
		
		double den = Math.sqrt(den1)*Math.sqrt(den2);
		
		if(den != 0)
		{
			return (num/den);
		}
		else
		{
			throw new Exception("denominator is 0");
		}
		

	}

	private static Integer[] getIntResponseArray(String[] responses) {
		Integer[] intArr = new Integer[responses.length-1];
		for(int i = 0 ; i < responses.length-1; i++)
		{
			intArr[i] = Integer.parseInt(responses[i]);
		}
		return intArr;
	}
	
}