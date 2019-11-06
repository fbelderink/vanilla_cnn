package io.fynn.neuralnetworks.layers.activations;

public class Activation{

    String activationfunction;

    public Activation(String activationfunction){
        this.activationfunction = activationfunction;
    }

    public float[][] feedfoward(float[][] X){
        switch(activationfunction){
            case "linear": return new Linear().feedfoward(X);
            case "softmax": return new Softmax().feedfoward(X);
            case "sigmoid": return new Sigmoid().feedfoward(X);
            case "tanH": return new TanH().feedfoward(X);
            case "relu": return new Relu().feedfoward(X);
        }

        return X;
    }

    public float[][][][] feedfoward(float[][][][] X){
        switch(activationfunction){
            case "linear": return new Linear().feedfoward(X);
            case "softmax": return new Softmax().feedfoward(X);
            case "sigmoid": return new Sigmoid().feedfoward(X);
            case "tanH": return new TanH().feedfoward(X);
            case "relu": return new Relu().feedfoward(X);
        }

        return X;
    }

    public float[][] backprop(float[][] X){
        
        switch(activationfunction){
            case "linear": return new Linear().backprop(X);
            case "softmax": return new Softmax().backprop(X);
            case "sigmoid": return new Sigmoid().backprop(X);
            case "tanH": return new TanH().backprop(X);
            case "relu": return new Relu().backprop(X);
        }

        return X;
    }

    public float[][][][] backprop(float[][][][] X){

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