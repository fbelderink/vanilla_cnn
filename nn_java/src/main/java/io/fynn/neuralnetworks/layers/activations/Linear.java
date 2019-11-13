package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.narray;

public class Linear{

    public narray feedfoward(narray X){
        return X;
    }

    public narray backprop(narray X){

        narray output = new narray(X.shape());
        
        output.setAll(1.0f);
        	
        return output;
    }

}