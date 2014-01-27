package cotraining;

import java.util.HashSet;
import java.util.Set;

import weka.core.Instances;

public class ClassInstances
{
	private Instances trainInstances;
	private Instances unlabeledInstances;
	private Instances testInstances;
	private Set<Integer> testIndices = new HashSet<Integer>();
	
	public ClassInstances()
	{}
	
	public ClassInstances(Instances trainInstances,
			Instances unlabeledInstances, Instances testInstances) {
		this.trainInstances = trainInstances;
		this.unlabeledInstances = unlabeledInstances;
		this.testInstances = testInstances;
	}
	
	public Instances getTestInstances() {
		return testInstances;
	}
	public void setTestInstances(Instances testInstances) {
		this.testInstances = testInstances;
	}
	public Instances getTrainInstances() {
		return trainInstances;
	}
	public void setTrainInstances(Instances trainInstances) {
		this.trainInstances = trainInstances;
	}
	public Instances getUnlabeledInstances() {
		return unlabeledInstances;
	}
	public void setUnlabeledInstances(Instances unlabeledInstances) {
		this.unlabeledInstances = unlabeledInstances;
	}

	public Set<Integer> getTestIndices() {
		return testIndices;
	}

	public void setTestIndices(Set<Integer> testIndices) {
		this.testIndices.addAll(testIndices);
	}
	
	
}