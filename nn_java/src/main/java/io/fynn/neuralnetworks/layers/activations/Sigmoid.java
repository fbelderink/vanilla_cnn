package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.narray;

public class Sigmoid{

    public narray feedfoward(narray X) throws Exception{

        narray output = new narray(X.shape());
        
        for(int i = 0; i < output.length(); i++){     
            output.set((float) (1.0f / (1.0f + Math.exp(-X.get1D(i)))), i,0);
        }

        return output;
    }

    public narray backprop(narray X) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            output.set((float) X.get1D(i) * (1.0f - X.get1D(i)), i,0); 
        }

        return output;
    }
}