package learn.nn.core;

import learn.math.util.VectorOps;

import java.util.Scanner;

/**
 * A LogisticUnit is a Unit that uses a sigmoid
 * activation function.
 */
public class LogisticUnit extends NeuronUnit {
	
	/**
	 * The activation function for a LogisticUnit is a 0-1 sigmoid
	 * centered at z=0: 1/(1+e^(-z)). (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		double denominator = 1.0+Math.exp(-z);
		return 1.0/denominator;
	}
	
	/**
	 * Derivative of the activation function for a LogisticUnit.
	 * For g(z)=1/(1+e^(-z)), g'(z)=g(z)*(1-g(z)) (AIMA p. 727).
	 * @see //calculus.subwiki.org/wiki/Logistic_function#First_derivative
	 */
	public double activationPrime(double z) {
		double y = activation(z);
		return y * (1.0 - y);
	}

	/**
	 * Update this unit's weights using the logistic regression
	 * gradient descent learning rule (AIMA Eq 18.8).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		//build temporary weight[]
		double[] weights = new double[incomingConnections.size()];

		//get weights
		for(int i=1;i<incomingConnections.size();i++){
			weights[i] = getWeight(i);
		}

		//for each weight, update accordingly
		for(int i=1; i<weights.length;i++){
			double activation = activation(VectorOps.dot1(weights,x));
			double q = alpha*(y-activation) * activation* (1.0-activation)* x[i-1];

			/*double activation = activation(getInputSum());
			double q = alpha*(y-activation) * activation* (1.0-activation)* x[i-1];*/

			weights[i] = weights[i] + q;
			setWeight(i,weights[i]);

			//setWeight(i,(weights[i]+q));

			//System.out.println("q= "+ q);
		}

		/*//DUBUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
		Scanner scr = new Scanner(System.in);//debugger
		System.out.println("exmInputs: ");
		for(double i: x){
			System.out.print(i+", ");
		}
		System.out.println("");
		System.out.println("weights:");
		for(double i: weights){
			System.out.print(i+", ");
		}
		System.out.println("");
		System.out.println(VectorOps.dot1(weights,x));
		System.out.println(activation(VectorOps.dot1(weights,x)));
		scr.nextLine();*/

	}
}
