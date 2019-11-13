package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.*;

public class Softmax{

    numpy np = new numpy();

    public narray feedfoward(narray X) throws Exception{

        narray output = new narray(X.shape());

        float max = np.max(X);
        float sum = 0.0f;
        for(int i = 0;i < X.length(); i++){
            sum += Math.exp(X.get1D(i) - max);
        }

        for(int i = 0; i < output.length(); i++){
            output.set((float) (Math.exp(X.get1D(i) - max) / sum), i);
        }

        return output;
    }

    public narray backprop(narray X) throws Exception{

        narray output = new narray(X.shape());

        for(int i = 0; i < output.length(); i++){
            output.set(X.get1D(i) * (1.0f - X.get1D(i)), i);
        }

        return output;
    }

}