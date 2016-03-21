package com.flipkart.fdp.ml.transformer;

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

@Getter
@Setter
public abstract class TransformerBase implements Transformer {
    private Set<String> inputKeys = new LinkedHashSet<>(Arrays.asList("input"));
    private String outputKey = "output";
}
