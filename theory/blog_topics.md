First thing I tried to create was a matrix class but then realised that if I wanted to make a more  general library then i need to learn about and implement TENSORS OH YEAHHH BABY TENSORS

TALK ABOUT HOW CONFUSING TENSORS ARE SINCE THEY (technically different from physics)

Talk about TENSORS
Talk about how FUCKING TENSORS work (broadcasting, operations) BAM (i have been watching too much statquest)

Talk About COMPUTATION GRAPHS, DOUBLE BAM

I know what you are thinking, but what about the backwards pass? (talk about manual gradient calculation)
Explain that is this infeasible for a large/complex NN/computation graph
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
A,B,D,E,D,F (it loops backwards)

Since the D tensor used to create F was not a unique tensor (A copy with no parent/backwards function) it was sorting the ordering incorrectly for the context

Solution: Khans algorithm (more complex)

GUESS WHAT? THE SOLUTION ISNT ACTUALLY A SOLUTION BECAUSE OF SHARED POINTERS...
It was creating a shared pointer from an instance of an object which doesnt replace reference with ashared pointer so basically, its shit
I am either going to have to rewrite everything (~1000 lines) to make use of shared pointers
Or i could make things raw pointers... which seems easier but is not the c++ way and could cause problems later on.

Also I need to make all of my functions return pointers to tensors because right now it copies the tensor each time and its just really bad code.

Yeah umm, according to this documentation: <https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.html>
i was using the bad design choice so i can only blame myself because it is in writing to not use that patten. I probably should not of made this my first c++ project. Well, i am extremely demotivated since all of the code i wrote need to be refactored to use shared pointers which is just a pain in the arse.

IDK WHY BUT I WANT TO LEARN HOW TO USE MANIM (CREATE COOL REALLY COOL MATH ANIMATIONS JUST THINK ABOUT HOW BROADCASTING WOULD LOOK ANIMATED, IN MY HEAD IT LOOKS DAMN COOL. I CAN JUST IMAGINE IT LIKE A MATRIX THEN ROTATED A BIT AND A COOL DUPLICATION ANIMATION (THE VIRTUAL NUMBERS WOULD BE IN LIKE A GRAYED OUT COLOR))

So i refactored everything and it seems to work but the code looks really bad.
Looking at how other frameworks do stuff, they create op-nodes which they house the backwards function so what i could do is perform a bunch of complex calculations then define an opnode that would do all of those things in one function instead of storing each.

Right now im using lambda functions which are not bad at all but they are not clean and modular so it needs to change but for right now im too lazy to fix it.
