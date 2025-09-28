# TODOs

1. Thread safety of GUI updates. The process_queues() method calls GUI methods from a background thread, but tkinter requires GUI updates on the main thread. GUI operations should use root.after() to schedule updates on the main thread.
1. Test of high-frequency GUI updates.
1. GUI Blocking - model loading takes time and freeze the GUI.
1. cleanup when stopping the pipeline
1. Queue Data Protocol - Queue items should have a defined structure like TypedDict for {'text': str} instead of assuming the format.
