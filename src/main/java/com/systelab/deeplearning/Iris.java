package com.systelab.deeplearning;

import lombok.Data;

@Data
public class Iris {

	private Double sepalLength;
	private Double sepalWidth;
	private Double petalLength;
	private Double petalWidth;

	private String irisClass;

	public Iris(Double sl, Double sw, Double pl, Double pw) {
		sepalLength = sl;
		sepalWidth = sw;
		petalLength = pl;
		petalWidth = pw;
	}

}
