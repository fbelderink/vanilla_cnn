package io.fynn.neuralnetworks.layers;

import io.fynn.neuralnetworks.numpy.narray;

public class Flatten extends Layer{

    narray input,output;

    @Override
    public narray feedforward(narray X){
        this.input = X;

        int a = 1;
        for(int i = 0; i < X.shape().length; i++){
            a *= X.shape(i);
        }

        this.output = new narray(X.getArray(), a,1);

        return this.output;
    }

    @Override
    public narray backprop(narray E,String loss,int layer_i) throws Exception{
        return E.reshape(this.input.shape());
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