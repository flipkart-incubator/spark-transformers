package com.flipkart.fdp.ml.modelinfo;

import com.flipkart.fdp.ml.transformer.CommonAddressFeaturesTransformer;
import com.flipkart.fdp.ml.transformer.Transformer;
import lombok.Data;

import java.util.HashSet;
import java.util.List;

@Data
public class CommonAddressFeaturesModelInfo extends AbstractModelInfo {
	private String mergedAddressParam;
	private String sanitizedAddressParam;

	private String numWordsParam;
	private String numCommasParam;
	private String numericPresentParam;
	private String addressLengthParam;
	private String favouredStartColParam;
	private String unfavouredStartColParam;

	private HashSet<String> favourableStarts;
	private HashSet<String> unFavourableStarts;

	@Override
	public Transformer getTransformer() {
		return new CommonAddressFeaturesTransformer(this);
	}
}
