package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.narray;

public class Relu{

    public narray feedfoward(narray X) throws Exception{
        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            if(X.get1D(i) <= 0){
                    output.set(0.0f, i);
            }
        }

        return output;
    }

    public narray backprop(narray X) throws Exception{
        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
                if(X.get1D(i) <= 0){
                    output.set(0.0f, i);
                }
                else{
                    output.set(1.0f, i);
                }
        }

        return output;    
    }
}