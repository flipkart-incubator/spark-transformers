package com.flipkart.fdp.ml.modelinfo;

import lombok.Getter;
import lombok.Setter;

import java.util.LinkedHashSet;
import java.util.Set;

@Getter
@Setter
public abstract class AbstractModelInfo implements ModelInfo {
    private Set<String> inputKeys = new LinkedHashSet<>();
    private Set<String> outputKeys = new LinkedHashSet<>();
}
