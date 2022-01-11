package learn.nn.core;

import learn.math.util.VectorOps;

/**
 * A PerceptronUnit is a Unit that uses a hard threshold
 * activation function.
 */
public class PerceptronUnit extends NeuronUnit {
	
	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold
	 * at z=0. (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		if(z >= 0){return 1.0;}
		else return 0.0;
	}
	
	/**
	 * Update this unit's weights using the Perceptron learning
	 * rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		//build temporary weight[]
		double[] weights = new double[incomingConnections.size()+1];

		for(int i=1;i<incomingConnections.size();i++){
				weights[i] = getWeight(i);
		}

		//TODO: bias weight update? how
		for(int i=0; i<weights.length;i++){
			weights[i] = weights[i] +
					alpha*(y-activation(VectorOps.dot(weights,x))) * x[i];
		}
	}
}
