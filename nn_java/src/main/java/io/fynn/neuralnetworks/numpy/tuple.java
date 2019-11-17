package io.fynn.neuralnetworks.numpy;

public class tuple<A,B>{

    A a;
    B b;

    public tuple(A a,B b){
        this.a = a;
        this.b = b;
    }

    public A getFirst(){
        return this.a;
    }

    public B getSecond(){
        return this.b;
    }

    public boolean equals(tuple<A,B> t){
        return this.getFirst().equals(t.getFirst()) && this.getSecond().equals(t.getSecond());
    }

    public String toString(){
        return "(" + a + "," + b + ")";
    }
}