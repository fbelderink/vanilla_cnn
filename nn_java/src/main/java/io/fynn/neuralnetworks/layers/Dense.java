package io.fynn.neuralnetworks.layers;

import io.fynn.neuralnetworks.numpy.*;
import io.fynn.neuralnetworks.layers.activations.*;

public class Dense{

    public static numpy np = new numpy();

    public static int nodes;
    public static float lr;
    public static String activation;
    public static Activation activation_function;

    public float[][] input;
    public float[][] output;
    public float[][] W;


    public Dense(int nodes,float lr,String activation){
        this.nodes = nodes;
        this.lr = lr;
        this.activation = activation;
        this.activation_function = new Activation(activation);
     }

    public Dense(int nodes,float lr){
        this(nodes,lr,"linear");
    }

    public float[][] setWeights(int x, int y){
        if(this.activation.equals("relu")){
            return np.normal(0.0f, (float) (Math.pow(2 /  x, 0.5)), x, y);
        }

        return np.normal(0.0f, (float) (Math.pow(2 / x, 0.5)), x, y);
    }

    public float[][] feedfoward(float[][] X){

        this.input = X;

        if(W == null){
            W = setWeights(this.nodes, X[0].length);
        }

        output = this.activation_function.feedfoward(np.dot(X, W));

        return output;
    }

    public void backprop(){

    }

}