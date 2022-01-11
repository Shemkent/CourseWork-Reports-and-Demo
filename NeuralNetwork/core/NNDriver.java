package learn.nn.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class NNDriver {

    //NNDriver.java [file name] [number of outputs] [single/multi layer] [k value] [alpha]
    // [alpha] is not optional; if using decaying schedule, enter 0.
    // optional arguments after [alpha]:
    //      [+nhiddens layer 1] [+nhiddens layer 2] [+nhiddens layer 3] ...
    public static void main(String[] args) throws FileNotFoundException {
        //File read
        File DataSet = new File(args[0]);
        Scanner scr = new Scanner(DataSet);

        //Building Training set
        List<learn.nn.core.Example> exampleTrain = new ArrayList<>();
        int noutput =  Integer.parseInt(args[1]);
        int attributeSize = 2; //# of attributes

        Map<String,Integer> labelMapping = new HashMap();
        Set<String> labelSet = new HashSet<>();

        //for all examples
        while(scr.hasNextLine()){
            //split into array by single comma
            String[] currExampleRaw = scr.nextLine().split(",");
            attributeSize = currExampleRaw.length-1;

            double[] outputs = new double[noutput];
            double[] attributes = new double[currExampleRaw.length-1];

            //iterative addition of elements, except the last one which is output
            for(int i=0; i<attributes.length; i++){
                String a = currExampleRaw[i];
                attributes[i] = Double.parseDouble(a);
            }

            //initialize outputs as 0
            for(int i=0; i<noutput; i++){
                outputs[i] = 0.0;
            }

            //reading raw value of output, i.e. last element of stream
            String thisLabel = currExampleRaw[currExampleRaw.length-1];
            if(!(labelSet.contains(thisLabel))){ //if havent encountered
                labelSet.add(thisLabel);
                labelMapping.put(thisLabel,labelSet.size()-1);
            }
            //System.out.println(labelMapping.toString());//debug

            int thisKey = labelMapping.get(thisLabel);
            outputs[thisKey] = 1.0;

            learn.nn.core.Example exm = new Example(attributes,outputs);
            exampleTrain.add(exm);
        }

        //BUILDING NN
        //instantiating
        SingleLayerFeedForwardNeuralNetwork SLNN = null;
        MultiLayerFeedForwardNeuralNetwork MLNN = null;
        InputUnit[] inputs = new InputUnit[attributeSize];

        //initializing input unit collection
        for(int i=0;i< inputs.length;i++){
            inputs[i] = new InputUnit();
        }
        final String SINGLE ="1",MULTI="2";

        //NN construction and training
        if(args[2].equals(SINGLE)){
            NeuronUnit[] outputs = new LogisticUnit[noutput];//default

            //initializing output unit collection
            for(int i=0;i< outputs.length;i++){
                LogisticUnit tempUnit = new LogisticUnit();
                //connecting inputunits, fully connected
                for(int j=0; j<inputs.length;j++){
                    new Connection(inputs[j],tempUnit,0.0);
                }
                outputs[i] = tempUnit;
            }

            //constructor called
            SLNN = new SingleLayerFeedForwardNeuralNetwork(inputs,outputs);
            int k = Integer.parseInt(args[3]);
            double alpha  = Double.parseDouble(args[4]);

            /*SLNN.train(exampleTrain,alpha);
            System.out.println(SLNN.test(exampleTrain));*/

            System.out.println(SLNN.kFoldCrossValidate(exampleTrain,k,alpha));

        //and for multilayer NN
        } else if(args[2].equals(MULTI)){
            //int[] nhidden construction
            int[] nhidden = new int[args.length-5];
            //loop through all supplied hidden layers - i.e. everything after args[3]
            int j=0;
            for(int i=5;i<args.length;i++){
                nhidden[j] = Integer.parseInt(args[i]);
                j++;
            }

            MLNN = new MultiLayerFeedForwardNeuralNetwork(attributeSize,nhidden,noutput);

            int k = Integer.parseInt(args[3]);
            double alpha  = Double.parseDouble(args[4]);

            /*System.out.print("enter desired epochNum: ");
            Scanner in = new Scanner(System.in);
            int epoch = in.nextInt();
            System.out.println(MLNN.kFoldCrossValidate(exampleTrain,k,epoch,alpha));
*/
            //System.out.println(MLNN.kFoldCrossValidate(exampleTrain,k,200,alpha));

            for(int e=0;e<600;e++){
                System.out.println(MLNN.kFoldCrossValidate(exampleTrain,k,e,alpha));
            }


        } else {System.out.println("Invalid Neural Network type"); return;}
    }
}
