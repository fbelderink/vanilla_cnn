package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.narray;

public class Relu extends Layer{

    boolean layer;
    narray input,output; 

    public Relu(boolean layer){
        this.layer = layer;
    }

    public Relu(){
        this(true);
    }

    public narray feedforward(narray X) throws Exception{
        this.input = X;

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            if(X.get1D(i) <= 0){
                    output.set(0.0f, i);
            }
        }

        this.output = output;

        return this.output;
    }

    public narray backprop(narray X,String loss,int layer_i) throws Exception{
        narray output = new narray(X.shape());

        if(!layer){
            for(int i = 0; i < output.length(); i++){
                if(X.get1D(i) <= 0){
                    output.set(0.0f, i);
                }
                else{
                    output.set(1.0f, i);
                }
            }
        }

        for(int i = 0; i < output.length(); i++){
            if(this.output.get1D(i) <= 0){
                output.set(0.0f, i);
            }else{
                output.set(X.get1D(i), i);
            }
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