package learn.nn.core;

import java.util.List;
import java.util.Scanner;

import learn.nn.core.Example;

/**
 * A SingleLayerFeedForwardNeuralNetwork is a single-layer feed-forward network
 * where all the inputs are directly connected to the outputs
 * (AIMA Section 18.7.2).
 */
public class SingleLayerFeedForwardNeuralNetwork extends FeedForwardNeuralNetwork {
	
	/**
	 * Construct and return a new SingleLayerFeedForwardNeuralNetwork with the given
	 * InputUnits for inputs and NeuronUnits for outputs. It's up
	 * to you to arrange the feed-forward connections between the Units
	 * properly.
	 */
	public SingleLayerFeedForwardNeuralNetwork(InputUnit[] inputs, NeuronUnit[] outputs) {
		super(new Unit[2][]);
		this.layers[0] = inputs;
		this.layers[1] = outputs;
	}
	
	/**
	 * Print this SingleLayerFeedForwardNeuralNetwork to stdout.
	 * We output the weights on the output units in tab-delimited format:
	 * UNITNUM w_0 w_1 ... w_n. 
	 */
	public void dump() {
		NeuronUnit[] outputs = this.getOutputUnits();
		for (int i=0; i < outputs.length; i++) {
			NeuronUnit unit = outputs[i];
			System.out.print(i);
			for (Connection conn : unit.incomingConnections) {
				System.out.format("\t%.2f", conn.weight);
			}
			System.out.println();
		}
	}
	
	/**
	 * Train this SingleLayerFeedForwardNeuralNetwork on the given Examples,
	 * using given learning rate alpha.
	 * This means updating the weights on the output units for
	 * each example on each step.
	 */
	public void train(List<Example> examples, double alpha) {
		for(Example i : examples){
			train(i,alpha);
		}
	}
	
	/**
	 * Train this SingleLayerFeedForwardNeuralNetwork on the given Example,
	 * using given learning rate alpha.
	 * This means updating the weights on the output units based on the example.
	 */
	public void train(Example example, double alpha) {
		NeuronUnit[] outputs = this.getOutputUnits();
		for (int j=0; j < outputs.length; j++) {
			outputs[j].update(example.inputs, example.outputs[j], alpha);
		}
	}
	
	/**
	 * Run a k-fold cross-validation experiment on this SingleLayerFeedForwardNeuralNetwork using
	 * the given Examples and return the average accuracy over the k trials.
	 */
	public double kFoldCrossValidate(List<Example> examples, int k, double alpha) {
		NeuralNetwork.Trainer trainer = new NeuralNetwork.Trainer() {
			public void train(NeuralNetwork network, List<Example> examples) {
				((SingleLayerFeedForwardNeuralNetwork)network).train(examples, alpha);
			}
		};
		NeuralNetwork.Tester tester = new NeuralNetwork.Tester() {
			public double test(NeuralNetwork network, List<Example> examples) {
				return ((SingleLayerFeedForwardNeuralNetwork)network).test(examples);
			}
		};
		return super.kFoldCrossValidate(examples, k, trainer, tester);
	}

	@Override
	public boolean test(Example example) {
		Scanner scr = new Scanner(System.in);//debugger
		double[] exmInputs = example.inputs;
		double[] exmOut = example.outputs;
		InputUnit[] inputUnits = this.getInputUnits();
		NeuronUnit[] outputUnits = this.getOutputUnits();

		//computing the model's output
		double[] thisOut = new double[example.outputs.length];

		//setting input
		for(int i=0;i<inputUnits.length;i++){
			inputUnits[i].setOutput(exmInputs[i]);
		}

		//computing output
		for(int i=0;i<outputUnits.length;i++){
			outputUnits[i].run();
			thisOut[i] = outputUnits[i].getOutput();
		}

		/*//DUBUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
			System.out.println("exmInputs: ");
			for(double i: exmInputs){
				System.out.print(i+", ");
			}
			System.out.println("");
			System.out.println("NNInputs:");
			for(InputUnit i:inputUnits){
				System.out.print(i.output+", ");
			}
			System.out.println("weights:");
			this.dump();
			System.out.println("exmOut:");
			for(double i: exmOut){
				System.out.print(i+", ");
			}
			System.out.println("");
			System.out.println("NNOut:");
			for(double i: thisOut){
				System.out.print(i+", ");
			}
		scr.nextLine();*/


		//System.out.println("thisin: "+inputs[0]+ " " + inputs[1] + " " + inputs[2]);//debug
		//System.out.println("thisout: "+thisOut[0]+ " " + thisOut[1] + " " + thisOut[2]);//debug
		//find the max output
		double max = 0; int index =0;
		for(int i=0;i<thisOut.length;i++){
			if(max < thisOut[i]){max = thisOut[i]; index =i;}
		}
		//ceiling on max, floor on else
		for(int i=0;i<thisOut.length;i++){
			thisOut[i] = 0.0;
		}
		thisOut[index] = 1.0;
		//System.out.println("thisout post-floor: "+thisOut[0]+ " " + thisOut[1] + " " + thisOut[2]);//debug

		//verifying result
		for(int i =0;i<exmOut.length;i++){
			//System.out.println("exm: "+ exmOut[i] +"; this: "+ thisOut[i]);//debug
			if(exmOut[i] != thisOut[i]){
				//System.out.println("wrong");
				return false;
			}
			//System.out.println();
		}
		//System.out.println("right");

		return true;
	}
}
