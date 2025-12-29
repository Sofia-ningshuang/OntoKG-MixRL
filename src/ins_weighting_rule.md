if class A to class B has a edge with weight 3,
then instance a (child of A) to instace b (child of B) should assign weight 3 to their connected edge and keep the same direction, i.e., A->B then a -> b;

if sub-class A' (child class of class A) to class B has a edge with weight 5,
then instance a (child of A' and grandchild of A) to instace b (child of B) should assign weight 5 to their connected edge and keep the same direction, i.e., A->B then a -> b;

when class A has no instances, use sub-class A' which has instances, to spreading weight to instance b, i.e., weight 5.