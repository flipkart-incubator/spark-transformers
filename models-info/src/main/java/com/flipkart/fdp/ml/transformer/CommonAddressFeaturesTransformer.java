package com.flipkart.fdp.ml.transformer;

import com.flipkart.fdp.ml.modelinfo.CommonAddressFeaturesModelInfo;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CommonAddressFeaturesTransformer implements Transformer {
	private CommonAddressFeaturesModelInfo modelInfo;
	private final Pattern pattern = Pattern.compile("[0-9]");

	public CommonAddressFeaturesTransformer(CommonAddressFeaturesModelInfo modelInfo) {
		this.modelInfo = modelInfo;
	}

	@Override
	public void transform(Map<String, Object> input) {
		String[] sanitizedAddress = (String[]) input.get(modelInfo.getSanitizedAddressParam());
		String mergedAddress = (String) input.get(modelInfo.getMergedAddressParam());

		input.put(modelInfo.getNumWordsParam(), (double)sanitizedAddress.length);
		input.put(modelInfo.getNumCommasParam(), (double)mergedAddress.split(",").length - 1);
		input.put(modelInfo.getNumericPresentParam(), getNumericPresent(sanitizedAddress));
		input.put(modelInfo.getAddressLengthParam(), getAddressLength(sanitizedAddress));
		input.put(modelInfo.getFavouredStartColParam(), getFavouredStartCol(sanitizedAddress));
		input.put(modelInfo.getUnfavouredStartColParam(), getUnfavouredStartCol(sanitizedAddress));
	}


	private double getAddressLength(String[] sanitizedAddress) {
		return StringUtils.join(sanitizedAddress, " ").length();
	}

	private double getFavouredStartCol(String[] sanitizedAddress) {
		return modelInfo.getFavourableStarts().contains(sanitizedAddress[0]) ? 1.0 : 0.0;
	}

	private double getUnfavouredStartCol(String[] sanitizedAddress) {
		return modelInfo.getUnFavourableStarts().contains(sanitizedAddress[0]) ? 1.0 : 0.0;
	}

	private double getNumericPresent(String[] sanitizedAddress) {

		String address = StringUtils.join(sanitizedAddress, " ");
		Matcher match = pattern.matcher(address);
		return match.find() ? 1 : 0;
	}

	@Override
	public Set<String> getInputKeys() {
		return modelInfo.getInputKeys();
	}

	@Override
	public Set<String> getOutputKeys() {
		return modelInfo.getOutputKeys();
	}
}
