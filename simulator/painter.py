"""
painter package
"""
import tkinter as tk

from .utils import parse_microbatch_key, parse_microbatch_key_Vshape


class SchedulingPainter:
    """Scheduling Painter"""

    def __init__(self, config: dict) -> None:
        self._pp_size = config["pp_size"]
        self._pp_height = config["pp_height"]
        self._pp_align = config["pp_align"]
        self._pixel_base = config["pixel_base"]
        self._forward_length = [_len * config["pixel_base"] for _len in config["forward_length"]]
        self._backward_length = [_len * config["pixel_base"] for _len in config["backward_length"]]
        self._weight_length = [_len * config["pixel_base"] for _len in config["weight_length"]] 
        
        self._tk_root = tk.Tk()
        self._tk_root.title("SchedulingPainter")

        self._highlight_state = {}
        self._item2color = {}
        self._item2block = {}
        self._item2mid = {}

    def _highlight_and_resume_block(self, canvas, item_id):
        if self._highlight_state[item_id]:
            self._highlight_state[item_id] = False
            canvas.itemconfig(item_id, fill=self._item2color[item_id])
        else:
            self._highlight_state[item_id] = True
            canvas.itemconfig(item_id, fill="yellow")

    def draw(self, data: dict) -> None:
        """draw with tkinter"""
        # Convert data offset to pixels
        data = {key: val * self._pixel_base for key, val in data.items()}
        max_key = max(data, key=data.get)
        _, max_key_pid, _ = parse_microbatch_key(max_key)

        canvas_width = data[max_key] + self._backward_length[max_key_pid] + 2 * self._pp_align
        canvas_height = (self._pp_height + self._pp_align) * self._pp_size

        # 0. Create label canvas
        label_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=30)
        y_label = (0 + 30) // 2 + 5

        label_canvas.create_text(self._pp_align + 55, y_label, text="MinExecutionTime:")
        label_canvas.create_text(
            self._pp_align + 127,
            y_label,
            text=f"{(data[max_key] + self._backward_length[max_key_pid])//self._pixel_base}",
        )

        label_canvas.create_text(
            canvas_width - self._pp_align - 137, y_label, text="CurrentBlockCoords:"
        )
        coords_label = label_canvas.create_text(
            canvas_width - self._pp_align - 37, y_label, text="(start,end)"
        )
        label_canvas.pack()

        # 1. Create main canvas
        main_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=canvas_height)
        main_canvas.pack()

        # 2. Add timeline for each pipeline
        for pid in range(self._pp_size):
            x0 = self._pp_align
            y0 = (self._pp_height + self._pp_align) * pid + 5
            x1 = canvas_width - self._pp_align
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            main_canvas.create_rectangle(x0, y0, x1, y1, outline="black")
     
        # 3. Draw execution block for each microbatch according to start and end time
        for microbatch_key, offset in data.items():
            flagnum, pid, mid = parse_microbatch_key(microbatch_key)

            x0 = self._pp_align + offset
            y0 = (self._pp_height + self._pp_align) * pid + 5
            addx = 0
            endstring = ''
            color = ''
            if flagnum == 1:
                addx = self._forward_length[pid]
                endstring = 'f'
                color = "#00FF7F"
            elif flagnum ==2:
                addx = self._backward_length[pid]
                endstring = 'b'
                color = "#00BFFF"
            elif flagnum == 3:
                addx = self._weight_length[pid]
                endstring = 'w'
                color = "#BBBF00"
            x1 = x0 + addx
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            tag = f"p_{pid}_m_{mid}_" + endstring

            block = main_canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags=tag)
            text = main_canvas.create_text(
                (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid+1}"
            )

            # print(f"block {tag}: {x0}, {y0}, {x1}, {y1}", flush=True)

            self._highlight_state[block] = False
            self._item2color[block] = color
            self._item2block[block] = block
            self._item2block[text] = block
            self._item2mid[block] = mid

        # Register hook for highlighting execution block of this microbatch
        def _trigger_hook(event):
            del event

            items = main_canvas.find_withtag("current")
            if len(items) == 0:
                return

            current_item = self._item2block[items[0]]
            if current_item not in self._highlight_state:
                return

            item_coords = main_canvas.coords(current_item)
            current_start = int(item_coords[0] - self._pp_align) // self._pixel_base
            current_end = int(item_coords[2] - self._pp_align) // self._pixel_base
            label_canvas.itemconfig(
                coords_label, text=f"({current_start},{current_end})"
            )

            tags = [
                f"p_{pid}_m_{self._item2mid[current_item]}_{fbw}"
                for pid in range(self._pp_size)
                for fbw in ("f", "b", "w")
            ]
            items_same_microbatch = []
            for tag in tags:
                found = main_canvas.find_withtag(tag)
                if len(found) != 0:
                    items_same_microbatch.append(found[0])

            for item in items_same_microbatch:
                self._highlight_and_resume_block(main_canvas, item)

        main_canvas.bind("<Button-1>", _trigger_hook)

        self._tk_root.mainloop()

class SchedulingPainterVshape:
    """Scheduling Painter"""

    def __init__(self, config: dict) -> None:
        self._pp_size = config["pp_size"]
        self._pp_height = config["pp_height"]
        self._pp_align = config["pp_align"]
        self._pixel_base = config["pixel_base"]
        self._forward_length1 = [_len * config["pixel_base"] for _len in config["forward_length1"]]
        self._backward_length1 = [_len * config["pixel_base"] for _len in config["backward_length1"]]
        self._weight_length1 = [_len * config["pixel_base"] for _len in config["weight_length1"]]
        self._forward_length2 = [_len * config["pixel_base"] for _len in config["forward_length2"]]
        self._backward_length2 = [_len * config["pixel_base"] for _len in config["backward_length2"]]
        self._weight_length2 = [_len * config["pixel_base"] for _len in config["weight_length2"]] 
        

        self._tk_root = tk.Tk()
        self._tk_root.title("SchedulingPainter")

        self._highlight_state = {}
        self._item2color = {}
        self._item2block = {}
        self._item2mid = {}

    def _highlight_and_resume_block(self, canvas, item_id):
        if self._highlight_state[item_id]:
            self._highlight_state[item_id] = False
            canvas.itemconfig(item_id, fill=self._item2color[item_id])
        else:
            self._highlight_state[item_id] = True
            canvas.itemconfig(item_id, fill="yellow")

    def draw(self, data: dict) -> None:
        """draw with tkinter"""
        # Convert data offset to pixels
        data = {key: val * self._pixel_base for key, val in data.items()}
        max_key = max(data, key=data.get)
        _, max_key_pid, _ = parse_microbatch_key_Vshape(max_key)

        canvas_width = data[max_key] + self._backward_length2[max_key_pid] + 2 * self._pp_align
        canvas_height = (self._pp_height + self._pp_align) * self._pp_size

        # 0. Create label canvas
        label_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=30)
        y_label = (0 + 30) // 2 + 5

        label_canvas.create_text(self._pp_align + 55, y_label, text="MinExecutionTime:")
        label_canvas.create_text(
            self._pp_align + 127,
            y_label,
            text=f"{(data[max_key] + self._backward_length2[max_key_pid])//self._pixel_base}",
        )

        label_canvas.create_text(
            canvas_width - self._pp_align - 137, y_label, text="CurrentBlockCoords:"
        )
        coords_label = label_canvas.create_text(
            canvas_width - self._pp_align - 37, y_label, text="(start,end)"
        )
        label_canvas.pack()

        # 1. Create main canvas
        main_canvas = tk.Canvas(self._tk_root, width=canvas_width, height=canvas_height)
        main_canvas.pack()

        # 2. Add timeline for each pipeline
        for pid in range(self._pp_size):
            x0 = self._pp_align
            y0 = (self._pp_height + self._pp_align) * pid + 5
            x1 = canvas_width - self._pp_align
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            main_canvas.create_rectangle(x0, y0, x1, y1, outline="black")
     
        # 3. Draw execution block for each microbatch according to start and end time
        for microbatch_key, offset in data.items():
            flagnum, pid, mid = parse_microbatch_key_Vshape(microbatch_key)

            x0 = self._pp_align + offset
            y0 = (self._pp_height + self._pp_align) * pid + 5
            addx = 0
            endstring = ''
            color = ''
            if flagnum == 1:
                addx = self._forward_length1[pid]
                endstring = 'f1'
                color = "#00BF00"

            elif flagnum ==2:
                addx = self._backward_length1[pid]
                endstring = 'b1'
                color = "#00BFFF"
            elif flagnum == 3:
                addx = self._weight_length1[pid]
                endstring = 'w1'
                color = "#FF0000"
            if flagnum == 4:
                addx = self._forward_length2[pid]
                endstring = 'f2'
                color = "#00BF00"
            elif flagnum ==5:
                addx = self._backward_length2[pid]
                endstring = 'b2'
                color = "#00BFFF"
            elif flagnum == 6:
                addx = self._weight_length2[pid]
                endstring = 'w2'
                color = "#FF0000"
            x1 = x0 + addx
            y1 = (self._pp_height + self._pp_align) * (pid + 1) - 5
            tag = f"p_{pid}_m_{mid}_" + endstring

            block = main_canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags=tag)
            if flagnum == 1 or flagnum == 2 or flagnum == 3:
                text = main_canvas.create_text(
                    (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid+1}"
                )
            elif flagnum == 4 or flagnum == 5 or flagnum == 6:
                text = main_canvas.create_text(
                    (x0 + x1) // 2, (y0 + y1) // 2, text=f"{mid+1}"+"*", font="Arial"  
                )

            # print(f"block {tag}: {x0}, {y0}, {x1}, {y1}", flush=True)

            self._highlight_state[block] = False
            self._item2color[block] = color
            self._item2block[block] = block
            self._item2block[text] = block
            self._item2mid[block] = mid

        # Register hook for highlighting execution block of this microbatch
        def _trigger_hook(event):
            del event

            items = main_canvas.find_withtag("current")
            if len(items) == 0:
                return

            current_item = self._item2block[items[0]]
            if current_item not in self._highlight_state:
                return

            item_coords = main_canvas.coords(current_item)
            current_start = int(item_coords[0] - self._pp_align) // self._pixel_base
            current_end = int(item_coords[2] - self._pp_align) // self._pixel_base
            label_canvas.itemconfig(
                coords_label, text=f"({current_start},{current_end})"
            )

            tags = [
                f"p_{pid}_m_{self._item2mid[current_item]}_{fbw}"
                for pid in range(self._pp_size)
                for fbw in ("f1", "b1", "w1", "f2", "b2", "w2")
            ]
            items_same_microbatch = []
            for tag in tags:
                found = main_canvas.find_withtag(tag)
                if len(found) != 0:
                    items_same_microbatch.append(found[0])

            for item in items_same_microbatch:
                self._highlight_and_resume_block(main_canvas, item)

        main_canvas.bind("<Button-1>", _trigger_hook)

        self._tk_root.mainloop()
