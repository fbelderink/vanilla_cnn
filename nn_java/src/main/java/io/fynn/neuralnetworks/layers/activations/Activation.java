package io.fynn.neuralnetworks.layers.activations;

import io.fynn.neuralnetworks.numpy.narray;

public class Activation{

    String activationfunction;

    public Activation(String activationfunction){
        this.activationfunction = activationfunction;
    }

    public narray feedfoward(narray X) throws Exception{
        switch(activationfunction){
            case "linear": return new Linear().feedfoward(X);
            case "softmax": return new Softmax().feedfoward(X);
            case "sigmoid": return new Sigmoid().feedfoward(X);
            case "tanH": return new TanH().feedfoward(X);
            case "relu": return new Relu().feedfoward(X);
        }

        return X;
    }

    public narray backprop(narray X) throws Exception{
        
        switch(activationfunction){
            case "linear": return new Linear().backprop(X);
            case "softmax": return new Softmax().backprop(X);
            case "sigmoid": return new Sigmoid().backprop(X);
            case "tanH": return new TanH().backprop(X);
            case "relu": return new Relu().backprop(X);
        }

        return X;
    }

}