package io.fynn.neuralnetworks.layers.loss;

import io.fynn.neuralnetworks.numpy.narray;

public class Loss{
    String loss;

    public Loss(String loss){
        this.loss = loss;
    }

    public narray process(narray targets, narray outputs) throws Exception{
        switch(this.loss){
            case "mse": return (targets.subtract(outputs.getArray())).multiply(-1);
            case "crossentropy": return (outputs.subtract(targets.getArray()));
            default: throw new Exception("Invalid loss!");
        }
    }  
}