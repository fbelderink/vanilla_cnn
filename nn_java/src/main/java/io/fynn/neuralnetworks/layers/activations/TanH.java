package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.layers.Layer;
import io.fynn.neuralnetworks.numpy.narray;

public class TanH extends Layer{

    boolean layer;
    narray input,output;

    public TanH(boolean layer){
        this.layer = layer;
    }

    public TanH(){
        this(true);
    }

    public narray feedforward(narray X) throws Exception{
        this.input = X;

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            output.set((float) (2.0f / (1.0f + Math.exp(-2 * X.get1D(i))) - 1), i);
        }

        this.output = output;

        return output;
    }

    public narray backprop(narray X,String loss,int layer_i) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            if(!layer){
                output.set((float) (1.0f - Math.pow(X.get1D(i),2)), i);
            }

            output.set((float) (X.get1D(i) * 1.0f - Math.pow(this.output.get1D(i), 2)),i);
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