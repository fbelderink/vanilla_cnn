package io.fynn.neuralnetworks.layers;

import io.fynn.neuralnetworks.numpy.*;

import io.fynn.neuralnetworks.layers.activations.*;

public class Dense extends Layer{

    public static numpy np = new numpy();

    public int nodes;
    public float lr;
    public String activation;
    public Activation activation_function;

    public narray input,W,output;

    public Dense(int nodes,float lr,String activation){
        this.nodes = nodes;
        this.lr = lr;
        this.activation = activation;
        this.activation_function = new Activation(activation,false);
     }

    public Dense(int nodes,float lr){
        this(nodes,lr,"linear");
    }

    public narray setWeights(int x, int y){
        if(this.activation.equals("relu")){
            return np.normal(0.0f, (float) (Math.pow(2.0f / (float) (x), 0.5)), x, y);
        }

        return np.normal(0.0f, (float) (Math.pow(1.0f / (float) (x), 0.5)), x, y);
    }

    @Override
    public narray feedforward(narray X) throws Exception{

        this.input = X;

        if(this.W == null){
            this.W = setWeights(this.nodes,X.shape(0));
        }

        this.output = this.activation_function.feedforward(np.dot(W,X));

        return this.output;
    }

    @Override
    public narray backprop(narray E, String loss, int layer_i) throws Exception{
        narray next_error = np.dot(np.transpose2D(this.W), E);

        if(loss.equals("crossentropy") && layer_i == 0){
            this.W = this.W.subtract(np.dot(E, np.transpose2D(this.input)).multiply(this.lr).getArray());
        }else{
            this.W = this.W.subtract(np.dot(E.multiply(this.activation_function.backprop(this.output,loss,layer_i).getArray()),np.transpose2D(this.input)).multiply(this.lr).getArray());
        }

        return next_error;
    }

    @Override
    public narray getInput(){
        return this.input;
    }

    public narray getWeigths(){
        return this.W;
    }

    @Override
    public narray getOutput(){
        return this.output;
    }

}