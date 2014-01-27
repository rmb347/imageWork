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

public class PrivatePublicSeg {
	private static HashMap<String, String> map = new HashMap<String, String>();
	public static void main(String args[])
	{

		String path = "/Users/Aurnob/Downloads/images/src/data/sample/rahul/september9/10000/";
		
		File testOpFile = new File(path+"10000_unique");
		FileInputStream inputStream;
		try {
			
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
				map.put(csv[0], csv[2]);
				
				url.delete(0, url.length() - 1);
			}

			int privateCount = 0;
			for(String val : map.keySet())
			{
				String privOrPub = map.get(val);
				privOrPub = privOrPub.trim();
				System.out.println(privOrPub);
				if(privOrPub.contains("private"))
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
