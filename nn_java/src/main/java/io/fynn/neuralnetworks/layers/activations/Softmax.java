package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.*;

public class Softmax extends Layer{

    numpy np = new numpy();
    
    boolean layer;
    narray input,output;

    public Softmax(boolean layer){
        this.layer = layer;
    }

    public Softmax(){
        this(true);
    }

    public narray feedforward(narray X) throws Exception{
        this.input = X;

        narray output = new narray(X.shape());

        float max = np.max(X);
        float sum = 0.0f;
        for(int i = 0;i < X.length(); i++){
            sum += Math.exp(X.get1D(i) - max);
        }

        for(int i = 0; i < output.length(); i++){
            output.set((float) (Math.exp(X.get1D(i) - max) / sum), i);
        }

        this.output = output;

        return this.output;
    }

    public narray backprop(narray X,String loss,int layer_i) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            if(!layer){
                output.set(X.get1D(i) * (1.0f - X.get1D(i)), i);
            }
            output.set(X.get1D(i) * this.output.get1D(i) * (1.0f - this.output.get1D(i)), i,0);
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