package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.narray;

public class Linear extends Layer{

    boolean layer;
    narray input,output;

    public Linear(boolean layer){
        this.layer = layer;
    }

    public Linear(){
        this(true);
    }

    public narray feedforward(narray X){
        this.input = X.clone();
        this.output = this.input;
        return this.input;
    }

    public narray backprop(narray X,String loss,int layer_i){

        if(layer){
            return X;
        }

        narray output = new narray(X.shape());
         
        output.setAll(1.0f);
        	
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