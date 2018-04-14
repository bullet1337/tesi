$(function() {
    $('#to_sentence').on('change input', function () {
        $('html, body').stop(true, true).animate({
            scrollTop: $($('.sentence')[parseInt($('#to_sentence').val()) || 0]).offset().top
        }, 200);
    });

    polarity_map = {
        'pos': ['SLIGHTLY_POSITIVE', 'VERY_POSITIVE', 'POSITIVE'],
        'neg': ['SLIGHTLY_NEGATIVE', 'VERY_NEGATIVE', 'NEGATIVE'],
        'neu': ['NEUTRAL']
    }

    last = null;
    last_fact = null;

    $('.fact').click(function() {
        if (last) {
            last.each(function (i, x) {
                $(x).removeClass();
            });
        }

        ids = $(this).children().map(function (i, x) { return $(x).attr('id'); });
        polarity = $(this).parent().attr('class').split(' ')[0];

        current_fact = this;;
        if (current_fact != last_fact) {
            last = $(this).parents('.sentence-row').find('.sentence-text').children();
            last.each(function (i, x) {
                $(x).removeClass();
                if ($.inArray($(x).attr('id'), ids) != -1) {
                    $(x).addClass(polarity);
                }
            });
            last_fact = current_fact;
        } else {
            last = null;
            last_fact = null;
        }
    });

    $('#is_all').click(function() {
        if ($(this).is(':checked')) {
            $('#interval').hide();
        } else {
            $('#interval').show();
        }
        calc_stats();
    })

    $(window).scroll(function(){
		if ($(this).scrollTop() > 100) {
			$('.scrollToTop').fadeIn();
		} else {
			$('.scrollToTop').fadeOut();
		}
	});

	$('.scrollToTop').click(function(){
		$('html, body').animate({scrollTop: 0}, 800);
		return false;
	});

	$('.e_check, .p_check, #from, #to').on('change input', calc_stats);

	function calc_stats() {
	    sentences = $('.sentence');
	    if (!$('#is_all').is(':checked')) {
	        from = parseInt($('#from').val()) || 0;
	        to = parseInt($('#to').val()) + 1 || sentences.length;
	        if (from < to) {
	            sentences = sentences.slice(from, to);
	        }
	    }

        for (polarity in polarity_map) {
            data = {'e': {0: 0, 1: 0}, 'p': {0: 0, 1: 0}};
            polarity_map[polarity].forEach(function (pol) {
                ['e', 'p'].forEach(function (r) {
                    checked = sentences.find('.' + pol + ' .' + r + '_check:checked').length;
                    data[r][1] += checked;
                    data[r][0] += sentences.find('.' + pol + ' .' + r + '_check').length - checked;
                });
            });
            ['e', 'p'].forEach(function (r) {
                $('#' + polarity + ' #' + r +'_pr').html((data[r][1] / (data[r][0] + data[r][1])).toFixed(2));
            });
        }
        ['e', 'p'].forEach(function (r) {
            $('#all #' + r +'_pr').html(
                ((parseFloat($('#pos' + ' #' + r + '_pr').html())
                    + parseFloat($('#neg' + ' #' + r + '_pr').html())) / 2).toFixed(2)
            );
        });
	}
});