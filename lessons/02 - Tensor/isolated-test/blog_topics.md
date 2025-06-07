First thing I tried to create was a matrix class but then realised that if I wanted to make a more  general library then i need to learn about and implement TENSORS OH YEAHHH BABY TENSORS

TALK ABOUT HOW CONFUSING TENSORS ARE SINCE THEY (technically different from physics)

Talk about TENSORS
Talk about how FUCKING TENSORS work (broadcasting, operations) BAM (i have been watching too much statquest)

Talk About COMPUTATION GRAPHS, DOUBLE BAM

I know what you are thinking, but what about the backwards pass? (talk about manual gradient calculation)
Explain that is this infeasble for a large/complex NN/computation graph
INTRODUCE AUTO GRAD SYSTEMS, TRIPLE BAM

Auto grad system  -> correct scheduling

Give example of incorrect:

       F
     /   \
    D     E
         /
       D
     /   \
    A     B

Original Algorithm for topological sorting was DFS which for this case outputted:
F,D,A,B,E

Since the D tensor used to create F was not a unique tensor (A copy with no parent/backwards function) it was sorting the ordering incorrectly for the context

Solution: Khans algorithm (more complex)

IDK WHY BUT I WANT TO LEARN HOW TO USE MANIM (CREATE COOL REALLY COOL MATH ANIMATIONS JUST THINK ABOUT HOW BROADCASTING WOULD LOOK ANIMATED, IN MY HEAD IT LOOKS DAMN COOL. I CAN JUST IMAGINE IT LIKE A MATRIX THEN ROTATED A BIT AND A COOL DUPLICATION ANIMATION (THE VIRTUAL NUMBERS WOULD BE IN LIKE A GRAYED OUT COLOR))
