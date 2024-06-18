/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package supervisedMB;

import java.util.Objects;

/**
 *
 * @author Georgios
 */
public class Comparison {

    private final boolean match;
    private final float prob;
    private final String entity1;
    private final String entity2;

    Comparison(boolean m, float p, String e1, String e2) {
        match = m;
        prob = p;
        entity1 = e1;
        entity2 = e2;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Comparison other = (Comparison) obj;
        if (this.entity1.equals(other.getEntity1())) {
            return false;
        }
        return this.entity2.equals(other.getEntity2());
    }

    public String getEntity1() {
        return entity1;
    }

    public String getEntity2() {
        return entity2;
    }

    public float getProb() {
        return prob;
    }
    
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 11 * hash + Objects.hashCode(this.entity1);
        hash = 11 * hash + Objects.hashCode(this.entity2);
        return hash;
    }
    
    public boolean isMatch() {
        return match;
    }
}