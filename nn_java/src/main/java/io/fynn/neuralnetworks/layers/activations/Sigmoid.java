package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.narray;

public class Sigmoid extends Layer{

    boolean layer;
    narray input,output;

    public Sigmoid(boolean layer){
        this.layer = layer;
    }

    public Sigmoid(){
        this(true);
    }

    public narray feedforward(narray X) throws Exception{
        this.input = X.clone();

        narray output = new narray(X.shape());
        
        for(int i = 0; i < output.length(); i++){     
            output.set((float) (1.0f / (1.0f + Math.exp(-X.get1D(i)))), i,0);
        }

        this.output = output.clone();

        return this.output;
    }

    public narray backprop(narray X,String loss,int layer_i) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            if(!layer){
                output.set((float) X.get1D(i) * (1.0f - X.get1D(i)), i,0); 
            }else{
              output.set((float) X.get1D(i) * this.output.get1D(i) * (1.0f - this.output.get1D(i)), i,0);
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