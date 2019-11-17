package io.fynn.neuralnetworks.numpy;

import java.util.ArrayList;
import java.util.Arrays;

public class narray {

    float[] array;
    int[] shape;
    
    public narray(int... shape){
        this.array = new float[product(shape)];
        this.shape = shape;
    }

    public narray(float[] array,int... shape){
        this.array = array;
        this.shape = shape;
    }

    public float[] get(int... pointer) throws Exception{
        if(pointer.length > this.shape.length){
            throw new Exception("Can't compute shapes " + Arrays.toString(pointer) + " and " + Arrays.toString(this.shape) + "!");
        }
            
        int pos = 0;
        for(int i = 0; i < pointer.length; i++){
            int a = 1;
            for(int j = i + 1; j < this.shape.length; j++){
                a *= this.shape[j];
            }
            pos += a * pointer[i];
        }
        int to = 0;
        if(pointer.length == this.shape.length){
            to = pos + 1;
        }else{
            int a = 1;
            for(int i = pointer.length; i < this.shape.length; i++){
                a *= this.shape[i];
            }
            to = pos + a;
        }

        return Arrays.copyOfRange(this.array, pos, to);
        
    }

    public void set(float num,int... pointer) throws Exception{
        if(pointer.length != shape.length){
            throw new Exception("Can't compute shapes " + Arrays.toString(pointer) + " and " + Arrays.toString(this.shape) + "!");
        }

        int pos = 0;
        for(int i = 0; i < pointer.length; i++){
            int a = 1;
            for(int j = pointer.length - (i + 1); j >= 1; j--){
                a *= this.shape[j];
            }
            pos += a * pointer[i];
        }

        this.array[pos] = num;
    }

    public void setAll(float num){
        for(int i = 0; i < this.array.length; i++){
            this.array[i] = num;
        }
    }

    @SuppressWarnings("varargs")
    public narray slice(tuple<Integer,Integer>... fromToPairs) throws Exception{
        for(int i = 0; i < fromToPairs.length; i++){
            if(fromToPairs.length > this.shape.length){
                throw new Exception("Datatype not understood!");
            }
        }
        
        ArrayList<int[]> output = new ArrayList<>();
        int[] starting_positions = new int[this.array.length];
        for(int i = 0; i < this.array.length; i++){
            starting_positions[i] = i;
        }
        output.add(starting_positions);

        for(int i = 0; i < fromToPairs.length; i++){
            int sumshape = 1;
            for(int j = i + 1; j < this.shape.length;j++){
                sumshape *= this.shape[j];
            }
            ArrayList<int[]> next = new ArrayList<>();
            for(int l = 0; l < output.size(); l++){
                for(int k = fromToPairs[i].getFirst(); k < fromToPairs[i].getSecond(); k++){
                    next.add(Arrays.copyOfRange(output.get(l), k * sumshape, k * sumshape + sumshape));
                }
            }
            output = new ArrayList<int[]>(next);
            next.clear(); 
        }

        int[] shape = this.shape.clone();
        for(int i = 0; i < fromToPairs.length; i++){
            shape[i] = fromToPairs[i].getSecond() - fromToPairs[i].getFirst();
        }

        int len_output = 1;
        for(int i = 0; i < shape.length; i++){
            len_output *= shape[i];
        }
        
        float[] output_arr = new float[len_output];
        for(int i = 0; i < output.size(); i++){
            for(int j = 0; j < output.get(i).length; j++){
                output_arr[i * output.get(i).length + j] = this.array[output.get(i)[j]];
            }
        }

        return new narray(output_arr, shape);
    }

    @SuppressWarnings("varargs")
    public void setSlice(float num,tuple<Integer,Integer>... fromToPairs) throws Exception{
        for(int i = 0; i < fromToPairs.length; i++){
            if(fromToPairs.length > this.shape.length){
                throw new Exception("Datatype not understood!");
            }
        }

        ArrayList<int[]> output = new ArrayList<>();
        int[] starting_positions = new int[this.array.length];
        for(int i = 0; i < this.array.length; i++){
            starting_positions[i] = i;
        }
        output.add(starting_positions);

        for(int i = 0; i < fromToPairs.length; i++){
            int sumshape = 1;
            for(int j = i + 1; j < this.shape.length;j++){
                sumshape *= this.shape[j];
            }
            ArrayList<int[]> next = new ArrayList<>();
            for(int l = 0; l < output.size(); l++){
                for(int k = fromToPairs[i].getFirst(); k < fromToPairs[i].getSecond(); k++){
                    next.add(Arrays.copyOfRange(output.get(l), k * sumshape, k * sumshape + sumshape));
                }
            }
            output = new ArrayList<int[]>(next);
            next.clear(); 
        }
        
        for(int i = 0; i < output.size(); i++){
            for(int j = 0; j  < output.get(i).length; j++){
                this.array[output.get(i)[j]] = num;
            }
        }

    }

    @SuppressWarnings("varargs")
    public void setSlice(float num,int pos,tuple<Integer,Integer>... fromToPairs) throws Exception{
        for(int i = 0; i < fromToPairs.length; i++){
            if(fromToPairs.length > this.shape.length){
                throw new Exception("Datatype not understood!");
            }
        }

        ArrayList<int[]> output = new ArrayList<>();
        int[] starting_positions = new int[this.array.length];
        for(int i = 0; i < this.array.length; i++){
            starting_positions[i] = i;
        }
        output.add(starting_positions);

        for(int i = 0; i < fromToPairs.length; i++){
            int sumshape = 1;
            for(int j = i + 1; j < this.shape.length;j++){
                sumshape *= this.shape[j];
            }
            ArrayList<int[]> next = new ArrayList<>();
            for(int l = 0; l < output.size(); l++){
                for(int k = fromToPairs[i].getFirst(); k < fromToPairs[i].getSecond(); k++){
                    next.add(Arrays.copyOfRange(output.get(l), k * sumshape, k * sumshape + sumshape));
                }
            }
            output = new ArrayList<int[]>(next);
            next.clear(); 
        }
        int[] shape = this.shape.clone();
        for(int i = 0; i < fromToPairs.length; i++){
            shape[i] = fromToPairs[i].getSecond() - fromToPairs[i].getFirst();
        }

        int len_positions = 1;
        for(int i = 0; i < shape.length; i++){
            len_positions *= shape[i];
        }
        
        if(pos > len_positions){
            throw new Exception("Position not in range!");
        }

        int[] positions = new int[len_positions];
        for(int i = 0; i < output.size(); i++){
            for(int j = 0; j  < output.get(i).length; j++){
                positions[i * output.get(i).length + j] = output.get(i)[j];
            }
        }

        this.array[positions[pos]] = num;

    }

    public int[] shape(){
        return this.shape;
    }

    public int shape(int i){
        return this.shape[i];
    }

    public int length(){
        return this.array.length;
    }

    public float get1D(int i){
        return this.array[i];
    }

    public narray add(float... num) throws Exception{
        if(num.length != this.array.length && num.length != 1){
            throw new Exception("Can't compute shapes " + num.length + " and " + Arrays.toString(this.shape) + "!");
        }

        float[] output = new float[this.array.length];

        if(num.length == this.array.length){
            for(int i = 0; i < num.length; i++){
                output[i] = this.array[i] + num[i];
            }

            return new narray(output, this.shape); 
        }

        for(int i = 0; i < this.array.length; i++){
            output[i] = this.array[i] + num[0];
        }

        return new narray(output, this.shape);
    }

    public void setadd(int[] pointer,float... num) throws Exception{
        
        if(pointer.length > this.shape.length){
            throw new Exception("Can't compute shapes " + Arrays.toString(pointer) + " and " + Arrays.toString(this.shape) + "!");
        }

        int pos = 0;
        for(int i = 0; i < pointer.length; i++){
            int a = 1;
            for(int j = i + 1; j < this.shape.length; j++){
                a *= this.shape[j];
            }
            pos += a * pointer[i];
        }
        int to = 0;
        if(pointer.length == this.shape.length){
            to = pos + 1;
        }else{
            int a = 1;
            for(int i = pointer.length; i < this.shape.length; i++){
                a *= this.shape[i];
            }
            to = pos + a;
        }

        if(num.length != to - pos && num.length != 1){
            throw new Exception("Can't compute shapes " + num.length + " and " + (pos - to) + "!");
        }

        if(num.length == (to - pos)){
            for(int i = pos; i < to; i++ ){
                this.array[i] += num[i - pos];
            }
        }
        else{
            for(int i = pos; i < to; i++){
                this.array[i] += num[0];
            }
        }


    }

    public narray multiply(float... num) throws Exception{

        if(num.length != this.array.length && num.length != 1){
            throw new Exception("Can't compute shapes " + num.length + " and " + Arrays.toString(this.shape) + "!");
        }

        float[] output = new float[this.array.length];

        if(num.length == this.array.length){
            for(int i = 0; i < num.length; i++){
                output[i] = this.array[i] * num[i];
            }

            return new narray(output, this.shape); 
        }

        for(int i = 0; i < this.array.length; i++){
            output[i] = this.array[i] * num[0];
        }

        return new narray(output, this.shape);
    }

    public narray subtract(float... num) throws Exception{

        if(num.length != this.array.length && num.length != 1){
            throw new Exception("Can't compute shapes " + num.length + " and " + Arrays.toString(this.shape) + "!");
        }

        float[] output = new float[this.array.length];

        if(num.length == this.array.length){
            for(int i = 0; i < num.length; i++){
                output[i] = this.array[i] - num[i];
            }

            return new narray(output, this.shape);
        }

        for(int i = 0; i < this.array.length; i++){
            output[i] = this.array[i] - num[0];
        }

        return new narray(output, this.shape);
    }

    public narray divide(float... num) throws Exception{

        if(num.length != this.array.length && num.length != 1){
            throw new Exception("Can't compute shapes " + num.length + " and " + Arrays.toString(this.shape) + "!");
        }

        float[] output = new float[this.array.length];

        if(num.length == this.array.length){
            for(int i = 0; i < num.length; i++){
                output[i] = this.array[i] / num[i];
            }

            return new narray(output, this.shape);
        }

        for(int i = 0; i< this.array.length; i++){
            output[i] = this.array[i] / num[0];
        }

        return new narray(output, this.shape);
    }
    

    public int product(int[] shape){
        int output = 1;

        for(int i = 0; i < shape.length; i++){
            output *= shape[i];
        }

        return output;
    }

    public narray reshape(int... shape) throws Exception{
        int a = 1;
        for(int i = 0; i < shape.length; i++){
            a *= shape[i];
        }

        if(a != this.array.length){
            throw new Exception("Can't reshape " + Arrays.toString(this.shape) + " to " + Arrays.toString(shape));
        }

        this.shape = shape;

        return this;
    }

    public float[] getArray(){
        return this.array;
    }

    public String asString(){
        return Arrays.toString(this.array);
    }

    public narray clone(){
        return new narray(this.array,this.shape);
    }

}

