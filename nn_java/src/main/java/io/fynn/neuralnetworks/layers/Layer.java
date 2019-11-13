package io.fynn.neuralnetworks.layers;

import io.fynn.neuralnetworks.numpy.narray;

public abstract class Layer{

    public abstract narray feedforward(narray X) throws Exception;
    public abstract narray backprop(narray E, String loss, int layer_i) throws Exception;
    
    public abstract narray getInput();
    public abstract narray getOutput();
    public abstract int getNodes();

}