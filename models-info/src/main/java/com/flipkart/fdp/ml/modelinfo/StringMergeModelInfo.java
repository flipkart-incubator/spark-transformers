package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.StringMergeTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;

public class StringMergeModelInfo extends AbstractModelInfo{
	@Override
	public Transformer getTransformer() {
		return new StringMergeTransformer(this);
	}
}
