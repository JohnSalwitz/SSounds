# SSounds
Framework for Sound Recognition


### Notes
* This is early work on my concept around shop-tool recognition.   The big idea is to put a "listener" in my wood shop and have it identify which power tools I am using and when.  This has a couple purposes
  1. To publish (MQTT) a topic that would trigger a dust colleciton system (on tools like table saw, thickness planer, band saw)
  1. To catalog tool uses in some sort of calendar (just for kicks, I would like to see which tools I use when)

* sound_recorder.py -- sample app to test if mike is working (etc)
* sound_capture.py -- builds the "peaks.csv" file from recorded sounds.  This is input for ml classifier
* sound_test.py -- uses "peaks.csv" file to train classifer. then listens for sounds and guesses at classification (not tested on osx yet)


