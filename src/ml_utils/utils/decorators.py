from datetime import datetime
from functools import update_wrapper


def time_process(process_name: str = "Execution"):
    def wrap(func):
        def timer(*args, **kwargs):
            process_start_time = datetime.now()
            returned_obj = func(*args, **kwargs)
            process_end_time = datetime.now()

            run_time = str(process_end_time - process_start_time).split(":")

            print(
                "".join(
                    [
                        f"{process_name} Time: ",
                        f"{int(run_time[0])} hours " if int(run_time[0]) > 0 else "",
                        f"{int(run_time[1])} minutes " if int(run_time[1]) > 0 else "",
                        f"{round(float(run_time[2]))} seconds.",
                    ]
                )
            )

            return returned_obj

        return update_wrapper(timer, func)

    return wrap
