package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.narray;

public class Activation extends Layer{

    boolean layer;
    String activationfunction;
    narray input,output;

    public Activation(String activationfunction,boolean layer){
        this.activationfunction = activationfunction;
        this.layer = layer;
    }

    public Activation(String activationfunction){
        this(activationfunction, true);
    }

    @Override
    public narray feedforward(narray X) throws Exception{

        this.input = X;

        switch(activationfunction){
            case "linear": 
                this.output = new Linear().feedforward(X);
                break;
            case "softmax": 
                this.output = new Softmax().feedforward(X);
                break;
            case "sigmoid": 
                this.output = new Sigmoid().feedforward(X);
                break;
            case "tanH": 
                this.output = new TanH().feedforward(X);
                break;
            case "relu": 
                this.output = new Relu().feedforward(X);
                break;
            default:
                throw new Exception("There is no such activation function!");
        }

        return this.output;
    }

    @Override
    public narray backprop(narray X,String loss,int layer_i) throws Exception{
        narray output;
        switch(activationfunction){
            case "linear": 
                output = new Linear(this.layer).backprop(X,loss,layer_i);
                break;
            case "softmax": 
                output = new Softmax(this.layer).backprop(X,loss,layer_i);
                break;
            case "sigmoid": 
                output = new Sigmoid(this.layer).backprop(X,loss,layer_i);
                break;
            case "tanH": 
                output = new TanH(this.layer).backprop(X,loss,layer_i);
                break;
            case "relu": 
                output = new Relu(this.layer).backprop(X,loss,layer_i);
                break;
            default:
                throw new Exception("There is no such activation function!");
        }

        return output;
    }

    @Override
    public narray getInput(){
        return this.input;
    }

    @Override
    public narray getOutput(){
        return this.output;
    }

}