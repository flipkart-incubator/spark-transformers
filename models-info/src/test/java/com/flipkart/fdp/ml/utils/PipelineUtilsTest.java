package com.flipkart.fdp.ml.utils;

import com.flipkart.fdp.ml.transformer.Transformer;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PipelineUtilsTest {

    @Test
    public void testSingleTransformer() {
        Transformer t = createTransformer(Arrays.asList("a"), Arrays.asList("b"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t});
        assertEquals(inputs.size(), 1);
        assertTrue(inputs.containsAll(Arrays.asList("a")));
        assertFalse(inputs.contains("b"));
    }

    @Test
    public void testSingleTransformerWithColumnModification() {
        Transformer t = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t});
        assertEquals(inputs.size(), 3);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
    }

    @Test
    public void testIndependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("d", "e", "f"), Arrays.asList("d1", "e1", "f1"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2});
        assertEquals(inputs.size(), 6);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","d","e","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("d1"));
        assertFalse(inputs.contains("e1"));
        assertFalse(inputs.contains("f1"));
    }

    @Test
    public void testDependentTransformers() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a3", "a4"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(inputs.size(), 4);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("a2"));
        assertFalse(inputs.contains("c1"));
        assertFalse(inputs.contains("a3"));
        assertFalse(inputs.contains("a4"));
    }

    @Test
    public void testTransformersWithModification() {
        Transformer t1 = createTransformer(Arrays.asList("a", "b", "c"), Arrays.asList("a1", "b1", "c"));
        Transformer t2 = createTransformer(Arrays.asList("a1", "c", "f"), Arrays.asList("a2", "c1"));
        Transformer t3 = createTransformer(Arrays.asList("a", "a1", "a2"), Arrays.asList("a2"));
        Set<String> inputs = PipelineUtils.extractRequiredInputColumns(new Transformer[] {t1, t2, t3});
        assertEquals(inputs.size(), 4);
        assertTrue(inputs.containsAll(Arrays.asList("a","b","c","f")));
        assertFalse(inputs.contains("a1"));
        assertFalse(inputs.contains("b1"));
        assertFalse(inputs.contains("a2"));
        assertFalse(inputs.contains("c1"));
    }

    private Transformer createTransformer(final List<String> inputs, final List<String> outputs) {
        return new Transformer() {
            @Override
            public void transform(Map<String, Object> input) {
            }

            @Override
            public Set<String> getInputKeys() {
                return new HashSet<>(inputs);
            }

            @Override
            public Set<String> getOutputKeys() {
                return new HashSet<>(outputs);
            }
        };
    }
}
