package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.StringSanitizerTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;

public class StringSanitizerModelInfo extends AbstractModelInfo {
	@Override
	public Transformer getTransformer() {
		return new StringSanitizerTransformer(this);
	}
}
