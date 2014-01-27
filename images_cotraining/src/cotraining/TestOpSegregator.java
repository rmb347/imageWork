package cotraining;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import weka.classifiers.rules.Ridor;

public class TestOpSegregator {

	private static HashMap<String, String> map = new HashMap<String, String>();
	public static void main(String args[])
	{

		String path = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/september9/10000/properTest/";
		
		File testOpFile = new File(path+"testOp");
		FileInputStream inputStream;
		try {
			FileWriter fw1 = new FileWriter(path+"testop_class1");
			FileWriter fw2 = new FileWriter(path+"testop_class2");
			FileWriter fw3 = new FileWriter(path+"testop_combined");
			inputStream = new FileInputStream(testOpFile);
			DataInputStream in = new DataInputStream(inputStream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			StringBuffer url = new StringBuffer(2000);
			while ((url.append(br.readLine())).toString() != null) {
				if(url.toString().contains("null"))
				{
					break;
				}

				String[] csv = url.toString().split(",");
				map.put(csv[0], csv[1]);
				String prefix = csv[0] + " , " + csv[1] +", ";
				if(! csv[2].equals(csv[1]))
				{
					fw1.write(prefix + csv[2] +" , \n" );
				}
				if(! csv[3].equals(csv[1]))
				{
					fw2.write(prefix + csv[3] +" , \n" );
				}
				if(! csv[4].equals(csv[1]))
				{
					fw3.write(prefix + csv[4] +" , \n" );
				}
				url.delete(0, url.length() - 1);
			}
			fw1.close();
			fw2.close();
			fw3.close();
			int privateCount = 0;
			for(String val : map.keySet())
			{
				String privOrPub = map.get(val);
				privOrPub = privOrPub.trim();
				System.out.println(privOrPub);
				if(privOrPub.equals("private"))
				{
					privateCount++;
				}
			}
			System.out.println("Private Count - "+ privateCount);
			System.out.println("public Count - "+ (map.keySet().size() - privateCount));
		}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
