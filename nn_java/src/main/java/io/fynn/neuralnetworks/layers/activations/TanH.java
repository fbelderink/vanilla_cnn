package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.narray;

public class TanH{

    public narray feedfoward(narray X) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            output.set((float) (2.0f / (1.0f + Math.exp(-2 * X.get1D(i))) - 1), i);
        }

        return output;
    }

    public narray backprop(narray X) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            output.set((float) (1.0f - Math.pow(X.get1D(i),2)), i);
        }

        return output;
    }

}